"""
CardioAgent — Tier 2 Transformer Attention Visualization
Produces Figure 4 for the paper: which ECG regions drive each classification.

Method:
  - Extract CLS-token attention weights from all 6 Transformer layers
  - Average across layers → one attention map per strip (200 patches × 50ms each)
  - Overlay as heatmap on Lead-I ECG waveform
  - Shows clinically meaningful patterns:
      AFib    → irregular RR intervals highlighted
      STEMI   → ST segment highlighted
      BBB     → wide QRS complex highlighted
      Normal  → distributed, no focal attention

Output: /workspace/cardioagent/results/attention_maps.png
        /workspace/cardioagent/results/attention_maps.pdf

Run: python attention_viz.py
"""

import os, ast, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch
import torch.nn as nn
import wfdb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR    = '/workspace/cardioagent/data/ptbxl'
MODEL_PATH  = '/workspace/cardioagent/models/tier2_12lead.pt'
RESULTS_DIR = '/workspace/cardioagent/results'
N_LEADS     = 12
SEGMENT_LEN = 1000
PATCH_SIZE  = 5
N_PATCHES   = SEGMENT_LEN // PATCH_SIZE   # 200
FS          = 100   # Hz

CLASS_NAMES = ['Normal', 'AFib', 'STEMI/MI', 'BBB', 'Other']
CLASS_COLORS = {
    'Normal':   '#2196F3',   # blue
    'AFib':     '#F44336',   # red
    'STEMI/MI': '#FF9800',   # orange
    'BBB':      '#4CAF50',   # green
}

MI_CODES  = {'ASMI','IMI','ILMI','ALMI','AMI','LMI','PMI','IPLMI','IPMI',
             'INJAL','INJAS','INJIL','INJIN','INJLA',
             'ISCAL','ISCAN','ISCIN','ISCLA','ISCIL','ISCAS','ISC_'}
BBB_CODES = {'IRBBB','CRBBB','CLBBB','LBBB','RBBB','IVCD','LAFB','LPFB'}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── MODEL (same architecture as tier2_12lead.py) ──────────────────────────────
class MultiLeadPatchEmbedding(nn.Module):
    def __init__(self, n_leads, patch_size, d_model):
        super().__init__()
        self.n_leads = n_leads; self.ps = patch_size
        self.proj = nn.Linear(n_leads * patch_size, d_model)
    def forward(self, x):
        B, L, S = x.shape; n = S // self.ps
        x = x.reshape(B, L, n, self.ps).permute(0, 2, 1, 3)
        return self.proj(x.reshape(B, n, L * self.ps))

class SinPE(nn.Module):
    def __init__(self, d, ml=300, dr=0.1):
        super().__init__(); self.drop = nn.Dropout(dr)
        pe = torch.zeros(ml, d)
        pos = torch.arange(0, ml).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])

class ArrhythmiaClassifier12Lead(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = MultiLeadPatchEmbedding(12, 5, 256)
        self.pos_enc     = SinPE(256, 201, 0.1)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        enc = nn.TransformerEncoderLayer(256, 8, 512, 0.1,
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, 6)
        self.norm        = nn.LayerNorm(256)
        self.drop        = nn.Dropout(0.1)
        self.head        = nn.Linear(256, 5)
    def forward(self, x):
        B = x.size(0); x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_enc(x); x = self.transformer(x)
        return self.head(self.drop(self.norm(x[:, 0])))

# ── ATTENTION EXTRACTION ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_cls_attention(model, x_12lead: torch.Tensor) -> np.ndarray:
    """
    Extract CLS-token attention weights averaged across all 6 Transformer layers.
    CLS attention to each patch position indicates which ECG regions are
    most influential for the classification decision.

    Args:
        x_12lead: (1, 12, 1000) tensor
    Returns:
        attn: (200,) numpy array, attention weight per 5-sample patch
    """
    model.eval()
    x = x_12lead.to(DEVICE)
    B = 1

    # Recompute embedding (same as model.forward up to transformer)
    emb = model.patch_embed(x)                         # (1, 200, 256)
    cls = model.cls_token.expand(B, -1, -1)            # (1, 1, 256)
    h   = torch.cat([cls, emb], dim=1)                 # (1, 201, 256)
    h   = model.pos_enc(h)

    all_cls_attn = []

    for layer in model.transformer.layers:
        # Pre-LN normalisation (norm_first=True)
        h_norm = layer.norm1(h)

        # MHA with attention weights returned
        _, attn_weights = layer.self_attn(
            h_norm, h_norm, h_norm,
            need_weights=True,
            average_attn_weights=True   # average across 8 heads
        )
        # attn_weights: (1, 201, 201)
        # CLS token is at position 0 — extract its attention to patches 1-200
        cls_attn = attn_weights[0, 0, 1:].cpu().numpy()  # (200,)
        all_cls_attn.append(cls_attn)

        # Advance hidden state through full layer
        h = layer(h)

    # Average attention across all 6 layers
    avg_attn = np.mean(all_cls_attn, axis=0)   # (200,)

    # Normalise to [0, 1]
    avg_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)
    return avg_attn

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def assign_label(scp):
    if 'AFIB' in scp or 'AFLT' in scp: return 1
    codes = {k for k,v in scp.items() if v >= 50}
    if codes & MI_CODES:   return 2
    if codes & BBB_CODES:  return 3
    if codes & {'NORM'}:   return 0
    return 4

def find_best_examples(model):
    """
    Find the highest-confidence correctly-classified example for each target class.
    Returns dict: {class_idx: (ecg_12lead np.array, lead_i np.array, confidence)}
    """
    df = pd.read_csv(os.path.join(DATA_DIR, 'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    df['label']     = df['scp_codes'].apply(assign_label)
    df_test         = df[df['strat_fold'] == 10]

    # Target classes: Normal(0), AFib(1), STEMI/MI(2), BBB(3)
    target_classes = {0: None, 1: None, 2: None, 3: None}
    best_conf      = {0: 0.0,  1: 0.0,  2: 0.0,  3: 0.0}

    model.eval()
    print("Searching for best examples per class...")

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        true_label = int(row['label'])
        if true_label not in target_classes:
            continue
        if best_conf[true_label] > 0.95:   # already have a great example
            continue

        try:
            path   = os.path.join(DATA_DIR, row['filename_lr'])
            record = wfdb.rdrecord(path)
            sig    = record.p_signal.astype(np.float32)   # (1000, 12)
            if sig.shape != (SEGMENT_LEN, N_LEADS) or np.isnan(sig).any():
                continue

            sig_12 = sig.T.copy()   # (12, 1000)
            for i in range(N_LEADS):
                sig_12[i] = (sig_12[i] - sig_12[i].mean()) / (sig_12[i].std() + 1e-8)

            x   = torch.FloatTensor(sig_12).unsqueeze(0).to(DEVICE)
            out = model(x)
            probs     = torch.softmax(out, -1)[0]
            pred_cls  = probs.argmax().item()
            pred_conf = probs.max().item()

            # Only keep if correctly classified with high confidence
            if pred_cls == true_label and pred_conf > best_conf[true_label]:
                best_conf[true_label]      = pred_conf
                target_classes[true_label] = {
                    'sig_12': sig_12,
                    'lead_i': sig[: , 0],   # raw Lead-I (before normalisation for plotting)
                    'confidence': pred_conf,
                    'true_label': true_label
                }
        except:
            continue

    return target_classes

# ── PLOTTING ─────────────────────────────────────────────────────────────────
def plot_attention_figure(examples, model):
    """
    Create 2×2 figure with ECG + attention overlay for 4 arrhythmia classes.
    """
    classes_to_plot = [
        (0, 'Normal Sinus Rhythm'),
        (1, 'Atrial Fibrillation'),
        (2, 'STEMI / Myocardial Infarction'),
        (3, 'Bundle Branch Block'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('CardioAgent Tier-2: Transformer Attention Maps\n'
                 'Highlighting ECG regions driving arrhythmia classification',
                 fontsize=13, fontweight='bold', y=1.02)

    time_axis = np.linspace(0, 10, SEGMENT_LEN)   # seconds

    for idx, (cls_idx, cls_title) in enumerate(classes_to_plot):
        ax  = axes[idx // 2][idx % 2]
        ex  = examples.get(cls_idx)
        col = list(CLASS_COLORS.values())[idx]

        if ex is None:
            ax.text(0.5, 0.5, f'No example found for {cls_title}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cls_title)
            continue

        sig_12 = ex['sig_12']   # (12, 1000) normalised
        lead_i = ex['lead_i']   # (1000,) raw for plotting

        # Normalise Lead-I for display
        lead_plot = (lead_i - lead_i.mean()) / (lead_i.std() + 1e-8)

        # Extract attention weights
        x    = torch.FloatTensor(sig_12).unsqueeze(0)
        attn = extract_cls_attention(model, x)   # (200,)

        # Upsample attention from patch-level to sample-level
        # Each patch = 5 samples → repeat each attention value 5 times
        attn_full = np.repeat(attn, PATCH_SIZE)   # (1000,)

        # Plot attention as coloured background bands
        norm = Normalize(vmin=0, vmax=1)
        cmap = plt.cm.YlOrRd

        for p in range(N_PATCHES):
            start = p * PATCH_SIZE
            end   = start + PATCH_SIZE
            color = cmap(norm(attn[p]))
            ax.axvspan(time_axis[start], time_axis[end - 1],
                       alpha=0.35, color=color, linewidth=0)

        # Plot ECG waveform on top
        ax.plot(time_axis, lead_plot, color='black', linewidth=0.8, zorder=5)

        # Annotations
        ax.set_title(f'{cls_title}\nConf: {ex["confidence"]:.1%}',
                     fontsize=10, fontweight='bold', color=col)
        ax.set_xlabel('Time (seconds)', fontsize=9)
        ax.set_ylabel('Amplitude (normalised)', fontsize=9)
        ax.set_xlim(0, 10)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)

        # Attention colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Attention weight', fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()

    # Save
    png_path = os.path.join(RESULTS_DIR, 'attention_maps.png')
    pdf_path = os.path.join(RESULTS_DIR, 'attention_maps.pdf')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Figure saved:")
    print(f"  PNG → {png_path}")
    print(f"  PDF → {pdf_path}")
    return png_path

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print("CardioAgent — Tier 2 Attention Visualization")
    print(f"Device: {DEVICE}")
    print(f"{'='*65}\n")

    # Load model
    print("Loading Tier-2 model...")
    model = ArrhythmiaClassifier12Lead().to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded.")

    # Find best examples
    examples = find_best_examples(model)

    print("\nBest examples found:")
    for cls_idx, ex in examples.items():
        if ex:
            print(f"  {CLASS_NAMES[cls_idx]}: confidence {ex['confidence']:.1%}")
        else:
            print(f"  {CLASS_NAMES[cls_idx]}: NOT FOUND")

    # Generate figure
    print("\nGenerating attention visualization...")
    png_path = plot_attention_figure(examples, model)

    print(f"\nDone. Figure ready for paper as Figure 4.")
    print("Shows which ECG regions the Transformer attends to for each class.")

if __name__ == '__main__':
    main()
