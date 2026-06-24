"""
CardioAgent — Cross-Dataset Evaluation on Chapman-Shaoxing
Tier-2 (trained on PTB-XL) tested on Chapman-Shaoxing.

Chapman uses:
  - WFDB .hea + .mat format (not .dat)
  - SNOMED-CT codes in header "Dx:" field
  - Records nested: WFDBRecords/XX/XXX/JSXXXXX.hea
"""

import os, math, warnings, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, classification_report, f1_score
import wfdb
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
CHAP_DIR    = '/workspace/cardioagent/data/chapman'
T2_MODEL    = '/workspace/cardioagent/models/tier2_12lead.pt'
SEGMENT_LEN = 1000
N_LEADS     = 12
BATCH_SIZE  = 128
CLASS_NAMES = ['Normal', 'AFib', 'STEMI/MI', 'BBB', 'Other']

# Chapman SNOMED-CT code mapping
SNOMED_AFIB  = {'164890007', '164889003', '195080001'}
SNOMED_MI    = {'164865005', '57054005', '413444003', '164861001',
                '425419005', '426749004'}
SNOMED_BBB   = {'713427006', '713426002', '445118002', '251120003',
                '164909002', '59118001', '54016002'}
SNOMED_NORM  = {'426783006', '270492004'}

def parse_snomed_dx(hea_path):
    """Extract SNOMED codes from Chapman .hea header Dx field."""
    try:
        with open(hea_path, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') and 'Dx:' in line:
                    dx_part = line.split('Dx:')[-1].strip()
                    codes   = set(c.strip() for c in dx_part.split(','))
                    return codes
    except:
        pass
    return set()

def assign_chapman_label(codes):
    if codes & SNOMED_AFIB:  return 1
    if codes & SNOMED_MI:    return 2
    if codes & SNOMED_BBB:   return 3
    if codes & SNOMED_NORM:  return 0
    return 4

def load_chapman_data():
    """Recursively find all Chapman records and load 12-lead waveforms."""
    # Find all .hea files recursively
    hea_files = glob.glob(
        os.path.join(CHAP_DIR, 'WFDBRecords', '**', '*.hea'),
        recursive=True
    )
    print(f"Found {len(hea_files)} .hea files in Chapman")

    if len(hea_files) == 0:
        # Try flat directory
        hea_files = glob.glob(os.path.join(CHAP_DIR, '**', '*.hea'), recursive=True)
        print(f"Retry: found {len(hea_files)} .hea files")

    X, y = [], []
    label_dist = [0] * 5

    for hea_path in tqdm(hea_files, desc="Loading Chapman"):
        try:
            # Parse SNOMED codes from header
            codes = parse_snomed_dx(hea_path)
            label = assign_chapman_label(codes)

            # Load waveform (wfdb handles .mat automatically)
            rec_path = hea_path.replace('.hea', '')
            record   = wfdb.rdrecord(rec_path)
            sig      = record.p_signal

            if sig is None or sig.shape[0] < SEGMENT_LEN:
                continue

            # Crop/pad to SEGMENT_LEN
            sig = sig[:SEGMENT_LEN, :]         # (1000, n_leads)

            # Handle records with fewer than 12 leads
            if sig.shape[1] < N_LEADS:
                pad = np.zeros((SEGMENT_LEN, N_LEADS - sig.shape[1]), dtype=np.float32)
                sig = np.concatenate([sig, pad], axis=1)
            elif sig.shape[1] > N_LEADS:
                sig = sig[:, :N_LEADS]

            sig = sig.T.astype(np.float32)     # (12, 1000)

            # Normalise per lead
            for i in range(N_LEADS):
                std = sig[i].std()
                if std > 1e-8:
                    sig[i] = (sig[i] - sig[i].mean()) / std

            if np.isnan(sig).any() or np.isinf(sig).any():
                continue

            X.append(sig)
            y.append(label)
            label_dist[label] += 1

        except Exception:
            continue

    print(f"\nChapman loaded: {len(X)} records")
    print("Class distribution:")
    for i, (name, n) in enumerate(zip(CLASS_NAMES, label_dist)):
        print(f"  {i}={name}: {n:,}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# ── Model (same architecture as tier2_12lead.py) ──────────────────────────────
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
        B = x.size(0)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.head(self.drop(self.norm(x[:, 0])))

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    loader = DataLoader(ECGDataset(X, y), BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)
    preds, labels, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE); out = model(xb)
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(yb.tolist())
        probs.extend(torch.softmax(out, -1).cpu().tolist())
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    try:    auroc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except: auroc = 0.0
    return f1, auroc, preds, labels

def main():
    print(f"\n{'='*65}")
    print("CardioAgent — Cross-Dataset Evaluation: Chapman-Shaoxing")
    print("Tier-2 trained on PTB-XL (EU) → tested on Chapman (East Asia)")
    print(f"{'='*65}\n")

    X, y = load_chapman_data()

    if len(X) == 0:
        print("No Chapman records loaded. Check WFDBRecords directory.")
        return

    print(f"\nLoading Tier-2 model...")
    model = ArrhythmiaClassifier12Lead().to(DEVICE)
    ckpt  = torch.load(T2_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print("Model loaded. Running evaluation...")

    f1, auroc, preds, labels = evaluate(model, X, y)

    print(f"\n{'='*65}")
    print("Cross-Dataset Results — Chapman-Shaoxing")
    print(f"{'='*65}")
    print(classification_report(labels, preds,
                                  target_names=CLASS_NAMES, digits=4))
    print(f"Macro F1    : {f1:.4f}")
    print(f"Macro AUROC : {auroc:.4f}")
    print(f"\nIn-distribution  (PTB-XL):  AUROC = 0.8761")
    print(f"Out-of-distribution (Chapman): AUROC = {auroc:.4f}")
    print(f"Cross-dataset AUROC drop     : {0.8761 - auroc:.4f}")

    with open('/workspace/cardioagent/cross_dataset_results.json', 'w') as f:
        json.dump({'ptbxl_auroc': 0.8761, 'chapman_auroc': auroc,
                   'f1': f1, 'drop': 0.8761 - auroc}, f, indent=2)
    print(f"\nSaved → /workspace/cardioagent/cross_dataset_results.json")

if __name__ == '__main__':
    main()
