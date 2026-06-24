"""
CardioAgent — Tier 3: Temporal Aggregator
Uses REAL long-term ECG data from the LTAF Database (PhysioNet).

LTAF Database: 84 long-term (24-84 hr) ECG recordings with beat-level
AF annotations. This gives us genuine rhythm-burden ground truth —
not synthetic data.

Pipeline:
  1. Download LTAF Database from PhysioNet
  2. Segment each recording into 10-second strips → assign strip-level AF label
  3. Compute ground-truth 5-dim rhythm-burden vector from REAL annotations
  4. Add Tier-2-style noise to simulate classifier errors in deployment
  5. Train BiLSTM on (noisy_strip_sequence → burden_vector)

Output dims:
  [0] af_burden_pct       (0-1)
  [1] longest_episode_min normalised
  [2] episode_count       normalised
  [3] nocturnal_ratio     (0-1)
  [4] trend_slope         normalised

Run: python tier3_train.py
"""

import os, math, warnings, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import wfdb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FS              = 128          # LTAF sampling rate (Hz)
STRIP_SAMPLES   = FS * 10      # 10-second strip = 1280 samples
STRIP_MIN       = 10 / 60      # minutes per strip
WINDOW_STRIPS   = 144          # training window = 144 strips = 24 min
NOCTURNAL_START = 22
NOCTURNAL_END   = 6

INPUT_DIM       = 8            # [one_hot(5), confidence, time_sin, time_cos]
HIDDEN_DIM      = 128
N_LAYERS        = 2
DROPOUT         = 0.3
OUTPUT_DIM      = 5

BATCH_SIZE      = 64
EPOCHS          = 40
LR              = 1e-3
WEIGHT_DECAY    = 1e-4

DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
LTAF_DIR        = '/workspace/cardioagent/data/ltafdb'
MODEL_PATH      = '/workspace/cardioagent/models/tier3_aggregator.pt'
SEED            = 42

DIM_NAMES = ['AF Burden', 'Longest Ep', 'Episode Cnt', 'Nocturnal', 'Trend Slope']

# Simulated Tier-2 noise per class (from Tier-1 confusion matrix)
NOISE_RATES = {0: 0.05, 1: 0.10, 2: 0.08, 3: 0.07, 4: 0.05}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD LTAF DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def download_ltaf():
    os.makedirs(LTAF_DIR, exist_ok=True)
    # Check if already downloaded
    existing = [f for f in os.listdir(LTAF_DIR) if f.endswith('.hea')]
    if len(existing) >= 10:
        print(f"LTAF already downloaded ({len(existing)} records found), skipping.")
        return
    print("Downloading LTAF Database from PhysioNet (~500 MB)...")
    print("84 long-term ECG recordings, 24-84 hours each, real AF annotations")
    wfdb.dl_database('ltafdb', dl_dir=LTAF_DIR)
    print("LTAF download complete.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT STRIP-LEVEL LABELS FROM LTAF ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_ltaf_records():
    """List all valid LTAF records."""
    records = sorted({
        f.replace('.hea', '')
        for f in os.listdir(LTAF_DIR)
        if f.endswith('.hea')
    })
    return records


def extract_strip_labels(record_name: str) -> tuple:
    """
    Load a LTAF record and extract strip-level AF labels.

    LTAF annotations use rhythm labels:
      'N' or '(N'  = Normal sinus rhythm
      'AFIB' or '(AFIB' = Atrial Fibrillation
      'AFL'  = Atrial Flutter (treated as AFib for our 5-class taxonomy)

    Returns:
        strip_labels   : (n_strips,) int array — 0=Normal, 1=AFib, 4=Other
        start_hour     : float — recording start hour (0-24, estimated)
    """
    path = os.path.join(LTAF_DIR, record_name)
    try:
        record = wfdb.rdrecord(path)
        annot  = wfdb.rdann(path, 'atr')
    except Exception as e:
        return None, None

    sig_len    = record.sig_len
    n_strips   = sig_len // STRIP_SAMPLES
    if n_strips < WINDOW_STRIPS:
        return None, None

    # Build sample-level AF mask from annotations
    af_mask = np.zeros(sig_len, dtype=bool)
    rhythm  = 'N'
    for i, sample in enumerate(annot.sample):
        if sample >= sig_len:
            break
        aux = annot.aux_note[i].strip().replace('\x00', '') if annot.aux_note[i] else ''
        if aux:
            if 'AFIB' in aux or 'AFL' in aux:
                rhythm = 'AFIB'
            elif aux.startswith('(') or 'N' in aux:
                rhythm = 'N'
        if rhythm == 'AFIB' and sample < sig_len:
            # Mark from this sample until next annotation
            next_sample = annot.sample[i+1] if i+1 < len(annot.sample) else sig_len
            next_sample = min(next_sample, sig_len)
            af_mask[sample:next_sample] = True

    # Aggregate to strip level
    strip_labels = np.zeros(n_strips, dtype=int)
    for s in range(n_strips):
        start = s * STRIP_SAMPLES
        end   = start + STRIP_SAMPLES
        af_frac = af_mask[start:end].mean()
        strip_labels[s] = 1 if af_frac >= 0.5 else 0  # majority vote

    # Estimate start hour (LTAF has no absolute time, randomise for training)
    start_hour = random.uniform(0, 24)

    return strip_labels, start_hour


def load_all_ltaf_records():
    """
    Load all LTAF records and return list of (strip_labels, start_hour).
    """
    records = get_ltaf_records()
    print(f"Found {len(records)} LTAF records")

    all_sessions = []
    for rec in tqdm(records, desc="Extracting strip labels"):
        labels, start_hour = extract_strip_labels(rec)
        if labels is not None and len(labels) >= WINDOW_STRIPS:
            all_sessions.append((labels, start_hour))

    print(f"Usable records: {len(all_sessions)}")
    total_strips = sum(len(s[0]) for s in all_sessions)
    total_af     = sum((s[0] == 1).sum() for s in all_sessions)
    print(f"Total strips : {total_strips:,}")
    print(f"AF strips    : {total_af:,} ({100*total_af/total_strips:.1f}%)")
    return all_sessions

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COMPUTE RHYTHM-BURDEN VECTOR (from REAL labels)
# ─────────────────────────────────────────────────────────────────────────────
def compute_burden_vector(strip_labels: np.ndarray,
                          timestamps_h: np.ndarray) -> np.ndarray:
    """Compute 5-dim burden vector analytically from ground-truth strip labels."""
    n         = len(strip_labels)
    afib_mask = (strip_labels == 1)

    # [0] AF burden
    af_burden = float(afib_mask.mean())

    # [1] Longest episode (normalised)
    max_ep_min = 0.0
    ep_len     = 0
    for a in afib_mask:
        if a:
            ep_len    += 1
            max_ep_min = max(max_ep_min, ep_len * STRIP_MIN)
        else:
            ep_len = 0
    longest_ep_norm = max_ep_min / (n * STRIP_MIN + 1e-8)

    # [2] Episode count (normalised)
    ep_count = 0
    prev = False
    for a in afib_mask:
        if a and not prev:
            ep_count += 1
        prev = a
    ep_count_norm = ep_count / (n / 2.0 + 1e-8)

    # [3] Nocturnal ratio
    hour          = timestamps_h % 24
    nocturnal     = (hour >= NOCTURNAL_START) | (hour < NOCTURNAL_END)
    af_noc        = (afib_mask & nocturnal).sum()
    total_af      = afib_mask.sum()
    noc_ratio     = float(af_noc / (total_af + 1e-8)) if total_af > 0 else 0.0

    # [4] Trend slope (6 equal windows → linear regression)
    ws = max(1, n // 6)
    wb = [afib_mask[w*ws:(w+1)*ws].mean() for w in range(6)]
    x  = np.arange(6, dtype=float) - 2.5
    y  = np.array(wb) - np.mean(wb)
    slope = float((x * y).sum() / ((x * x).sum() + 1e-8))
    slope_norm = float(np.clip((slope + 0.5), 0.0, 1.0))

    return np.array([af_burden, longest_ep_norm, ep_count_norm,
                     noc_ratio, slope_norm], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — WINDOWED SESSION BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def add_tier2_noise(true_labels: np.ndarray) -> tuple:
    noisy = true_labels.copy()
    confs = np.ones(len(true_labels), dtype=np.float32)
    for i, lbl in enumerate(true_labels):
        if random.random() < NOISE_RATES.get(int(lbl), 0.05):
            wrong    = [c for c in range(5) if c != lbl]
            noisy[i] = random.choice(wrong)
            confs[i] = random.uniform(0.40, 0.64)
        else:
            confs[i] = random.uniform(0.70, 0.99)
    return noisy, confs


def strip_to_features(noisy_labels: np.ndarray,
                      confs: np.ndarray,
                      timestamps_h: np.ndarray) -> np.ndarray:
    one_hot = np.eye(5, dtype=np.float32)[noisy_labels]
    conf    = confs.reshape(-1, 1).astype(np.float32)
    angle   = 2 * math.pi * timestamps_h / 24
    t_sin   = np.sin(angle).reshape(-1, 1).astype(np.float32)
    t_cos   = np.cos(angle).reshape(-1, 1).astype(np.float32)
    return np.concatenate([one_hot, conf, t_sin, t_cos], axis=1)


def build_windows_from_ltaf(all_sessions: list,
                             windows_per_record: int = 50) -> tuple:
    """
    Slide a WINDOW_STRIPS window over each LTAF record.
    Each window becomes one training example.
    """
    X_list, y_list = [], []

    for strip_labels, start_hour in tqdm(all_sessions, desc="Building windows"):
        n = len(strip_labels)
        # Timestamps for this recording
        ts = np.array([start_hour + i * (STRIP_MIN / 60)
                       for i in range(n)]) % 24

        # Slide window with random stride to get diverse samples
        positions = sorted(random.sample(
            range(0, n - WINDOW_STRIPS),
            min(windows_per_record, n - WINDOW_STRIPS)
        ))

        for pos in positions:
            seg_labels = strip_labels[pos:pos + WINDOW_STRIPS]
            seg_ts     = ts[pos:pos + WINDOW_STRIPS]

            # Ground-truth burden from REAL annotations
            burden = compute_burden_vector(seg_labels, seg_ts)

            # Simulate Tier-2 noisy output
            noisy, confs = add_tier2_noise(seg_labels)
            features     = strip_to_features(noisy, confs, seg_ts)

            X_list.append(features)
            y_list.append(burden)

    X = np.stack(X_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    print(f"\nDataset built from REAL LTAF data:")
    print(f"  Windows : {len(X):,}  Shape: {X.shape}")
    print(f"  Burden vector mean per dim: {y.mean(axis=0).round(4)}")
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — DATASET + MODEL (same as before)
# ─────────────────────────────────────────────────────────────────────────────
class SessionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class TemporalAggregator(nn.Module):
    """
    Bidirectional LSTM — synthesises strip sequence -> 5-dim burden vector.
    Input  : (B, T, 8)
    Output : (B, 5)  in [0,1]

    POOLING_MODE = 'mean'      -> original mean pooling (baseline)
    POOLING_MODE = 'attention' -> learned attention pooling (Option 1 fix
                                   for Episode Count MAE target miss).
    Attention pooling lets the model weight strips near episode
    boundaries more heavily than uniformly-averaged strips, which
    mean pooling cannot do. This directly targets the documented
    weakness: mean pooling smooths over short bursts vs sustained
    episodes identically.
    """
    def __init__(self, pooling_mode='attention'):
        super().__init__()
        self.pooling_mode = pooling_mode
        self.input_proj = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.lstm = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, N_LAYERS,
                            batch_first=True, bidirectional=True,
                            dropout=DROPOUT if N_LAYERS > 1 else 0.0)
        lstm_out_dim = HIDDEN_DIM * 2

        if pooling_mode == 'attention':
            # Learned scalar attention score per timestep, softmax over time
            self.attn_proj = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_out_dim // 2, 1)
            )

        self.norm    = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(DROPOUT)
        self.head    = nn.Linear(lstm_out_dim, OUTPUT_DIM)

    def forward(self, x):
        x      = torch.relu(self.input_proj(x))
        out, _ = self.lstm(x)                       # (B, T, 2H)

        if self.pooling_mode == 'attention':
            attn_logits = self.attn_proj(out)        # (B, T, 1)
            attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T, 1)
            pooled = (out * attn_weights).sum(dim=1)  # (B, 2H)
        else:
            pooled = out.mean(dim=1)                  # (B, 2H) — baseline

        pooled = self.norm(pooled)
        return torch.sigmoid(self.head(self.dropout(pooled)))

    def predict_burden(self, features: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x   = torch.from_numpy(features).unsqueeze(0).to(
                      next(self.parameters()).device)
            out = self.forward(x)
        return out.squeeze(0).cpu().numpy()


class BurdenLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        p = pred   - pred.mean(0, keepdim=True)
        t = target - target.mean(0, keepdim=True)
        corr      = (p * t).sum(0) / ((p.pow(2).sum(0) * t.pow(2).sum(0)).sqrt() + 1e-8)
        corr_loss = 1.0 - corr.mean()
        return self.alpha * mse_loss + (1 - self.alpha) * corr_loss

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, total = 0.0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(X), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(y)
        total      += len(y)
    return total_loss / total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out  = model(X)
        total_loss += criterion(out, y).item() * len(y)
        preds.extend(out.cpu().numpy())
        targets.extend(y.cpu().numpy())
    n   = len(loader.dataset)
    p   = np.array(preds)
    t   = np.array(targets)
    mae = np.array([mean_absolute_error(t[:, d], p[:, d]) for d in range(OUTPUT_DIM)])
    return total_loss / n, mae

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print("CardioAgent — Tier 3: Temporal Aggregator")
    print("Training on REAL LTAF long-term AF recordings")
    print(f"Device: {DEVICE}")
    print(f"{'='*65}\n")

    os.makedirs('/workspace/cardioagent/models', exist_ok=True)

    download_ltaf()
    all_sessions = load_all_ltaf_records()

    if len(all_sessions) == 0:
        raise RuntimeError("No LTAF records loaded. Check download.")

    X, y = build_windows_from_ltaf(all_sessions, windows_per_record=60)

    # 80/10/10 split
    idx  = np.random.permutation(len(X))
    n_tr = int(0.80 * len(X))
    n_va = int(0.10 * len(X))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    print(f"Splits — Train:{len(tr):,}  Val:{len(va):,}  Test:{len(te):,}")

    train_loader = DataLoader(SessionDataset(X[tr], y[tr]),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(SessionDataset(X[va], y[va]),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(SessionDataset(X[te], y[te]),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    model     = TemporalAggregator().to(DEVICE)
    criterion = BurdenLoss(alpha=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler    = torch.cuda.amp.GradScaler()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"{'Ep':>4} | {'TrLoss':>8} | {'VaLoss':>8} | "
          + "  ".join(f"{n[:7]:>7}" for n in DIM_NAMES))
    print("─" * 80)

    best_mae = float('inf')
    for epoch in range(1, EPOCHS + 1):
        tr_loss         = train_epoch(model, train_loader, optimizer, criterion, scaler)
        va_loss, va_mae = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        mean_mae = va_mae.mean()
        marker   = " ✓" if mean_mae < best_mae else ""
        print(f"{epoch:>4} | {tr_loss:>8.5f} | {va_loss:>8.5f} | "
              + "  ".join(f"{m:>7.4f}" for m in va_mae) + marker)

        if mean_mae < best_mae:
            best_mae = mean_mae
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_mae': va_mae.tolist(),
                        'config': {'window_strips': WINDOW_STRIPS,
                                   'input_dim': INPUT_DIM,
                                   'dim_names': DIM_NAMES}},
                       MODEL_PATH)

    print(f"\n{'='*65}\nTest Set Evaluation\n{'='*65}")
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
    _, te_mae = eval_epoch(model, test_loader, criterion)

    print(f"\n  {'Dimension':<20} {'MAE':>8}  {'Target':>8}")
    print("  " + "─" * 40)
    targets = [0.05, 0.03, 0.02, 0.05, 0.04]
    for name, mae, tgt in zip(DIM_NAMES, te_mae, targets):
        print(f"  {name:<20} {mae:>8.4f}  {tgt:>8.4f}"
              + (" ✓" if mae <= tgt else " !"))
    print(f"\n  Mean MAE : {te_mae.mean():.4f}")
    print(f"  Model    → {MODEL_PATH}")
    print("Tier 3 complete. Next: python tier4_triage.py")

if __name__ == '__main__':
    main()
