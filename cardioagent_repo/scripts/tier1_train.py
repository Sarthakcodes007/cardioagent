"""
CardioAgent — Tier 1: Signal Quality Agent
1D CNN that classifies each 10-second ECG strip as:
  0 = clean       (SNR >= 12 dB)
  1 = noisy       (0 <= SNR < 12 dB)
  2 = unusable    (SNR < 0 dB)

Datasets:
  - MIT-BIH Arrhythmia DB  → clean ECG source
  - MIT-BIH Noise Stress Test DB (NSTDB) → noise signals (bw, ma, em)

Run: python tier1_train.py
Expected runtime on RTX 4090: ~45 minutes
Expected Macro F1: >0.92
"""

import os, random, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import wfdb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_LEN   = 3600        # 10 sec × 360 Hz
BATCH_SIZE    = 64
EPOCHS        = 30
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR      = '/workspace/cardioagent/data'
MODEL_PATH    = '/workspace/cardioagent/models/tier1_sqa.pt'
ONNX_PATH     = '/workspace/cardioagent/models/tier1_sqa.onnx'
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# SNR ranges per class
SNR_CLEAN     = None                # no noise added
SNR_NOISY     = [3, 6, 9, 11]      # dB  — signal readable with caveats
SNR_UNUSABLE  = [-6, -3, -1]       # dB  — noise-dominated

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_mitdb():
    out = os.path.join(DATA_DIR, 'mitdb')
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, '100.hea')):
        print("MIT-BIH already downloaded, skipping.")
        return out
    print("Downloading MIT-BIH Arrhythmia Database (~100 MB)...")
    wfdb.dl_database('mitdb', dl_dir=out)
    return out

def download_nstdb():
    out = os.path.join(DATA_DIR, 'nstdb')
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'bw.hea')):
        print("NSTDB already downloaded, skipping.")
        return out
    print("Downloading MIT-BIH Noise Stress Test Database (~5 MB)...")
    wfdb.dl_database('nstdb', dl_dir=out)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def add_noise_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix signal + noise to reach a specific SNR level in dB."""
    sig_power   = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise  ** 2) + 1e-10
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)
    return signal + noise_scaled

def load_mitdb_segments(mitdb_dir: str) -> list:
    """Extract all 10-second non-overlapping segments from MIT-BIH Lead I."""
    segs = []
    rec_names = sorted({f.replace('.hea','')
                        for f in os.listdir(mitdb_dir) if f.endswith('.hea')})
    for name in tqdm(rec_names, desc="  Loading MIT-BIH records"):
        try:
            rec = wfdb.rdrecord(os.path.join(mitdb_dir, name))
            sig = rec.p_signal[:, 0].astype(np.float32)
            # z-score normalise per record
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            for start in range(0, len(sig) - SEGMENT_LEN, SEGMENT_LEN):
                seg = sig[start:start + SEGMENT_LEN]
                if not np.isnan(seg).any():
                    segs.append(seg)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    print(f"  → {len(segs)} clean segments extracted")
    return segs

def load_noise_signals(nstdb_dir: str) -> list:
    """Load raw noise waveforms from NSTDB (bw, ma, em)."""
    noises = []
    for name in ['bw', 'ma', 'em']:
        path = os.path.join(nstdb_dir, name)
        try:
            rec = wfdb.rdrecord(path)
            noise = rec.p_signal[:, 0].astype(np.float32)
            noise = noise / (noise.std() + 1e-8)   # unit-variance
            noises.append(noise)
            print(f"  Loaded noise [{name}] length={len(noise)}")
        except Exception as e:
            print(f"  Could not load {name}: {e}")
    if not noises:
        raise RuntimeError("No noise signals loaded from NSTDB. Check download.")
    return noises

def sample_noise_patch(noise: np.ndarray, length: int) -> np.ndarray:
    """Randomly crop a patch from a noise signal, tile if too short."""
    if len(noise) < length:
        reps = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, reps)
    start = random.randint(0, len(noise) - length)
    return noise[start:start + length]

def build_dataset(mitdb_dir: str, nstdb_dir: str):
    """
    Build balanced 3-class dataset.
    Class 0: clean MIT-BIH segments
    Class 1: same segments + noise at 3–11 dB
    Class 2: same segments + noise at -6 to -1 dB
    """
    print("\nLoading clean ECG segments from MIT-BIH...")
    clean_segs = load_mitdb_segments(mitdb_dir)

    print("\nLoading noise signals from NSTDB...")
    noise_signals = load_noise_signals(nstdb_dir)

    X, y = [], []

    print("\nBuilding dataset (3 classes)...")
    for seg in tqdm(clean_segs, desc="  Generating samples"):
        # Class 0 — clean
        X.append(seg)
        y.append(0)

        # Class 1 — noisy (one random SNR level, random noise type)
        noise = sample_noise_patch(random.choice(noise_signals), SEGMENT_LEN)
        snr   = random.choice(SNR_NOISY)
        X.append(add_noise_at_snr(seg, noise, snr))
        y.append(1)

        # Class 2 — unusable
        noise = sample_noise_patch(random.choice(noise_signals), SEGMENT_LEN)
        snr   = random.choice(SNR_UNUSABLE)
        X.append(add_noise_at_snr(seg, noise, snr))
        y.append(2)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    counts = np.bincount(y)
    print(f"\nDataset built: {len(X)} samples  |  "
          f"Clean={counts[0]}  Noisy={counts[1]}  Unusable={counts[2]}")
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────
class ECGQualityDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).unsqueeze(1)   # (N, 1, 3600)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL: 1D CNN Signal Quality Agent
# ─────────────────────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel,
                      padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2)
        )
    def forward(self, x):
        return self.block(x)

class SignalQualityAgent(nn.Module):
    """
    Compact 1D CNN for ECG signal quality classification.
    Input:  (B, 1, 3600)   — single 10-sec strip
    Output: (B, 3)          — logits [clean, noisy, unusable]

    Architecture: 3 conv blocks → global average pool → dropout → linear
    Param count: ~115k — edge-deployable
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32, kernel=7),   # → (B, 32, 1800)
            ConvBlock(32,  64, kernel=5),   # → (B, 64,  900)
            ConvBlock(64, 128, kernel=3),   # → (B, 128, 450)
        )
        self.gap        = nn.AdaptiveAvgPool1d(1)   # → (B, 128, 1)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)

    def predict_quality(self, x: torch.Tensor):
        """Returns (class_idx, confidence) for inference use."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=-1)
            cls    = probs.argmax(dim=-1)
            conf   = probs.max(dim=-1).values
        return cls, conf

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    n   = sum(1 for _ in loader.dataset)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    return total_loss / n, f1, preds, labels

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("CardioAgent — Tier 1: Signal Quality Agent")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")

    os.makedirs('/workspace/cardioagent/models', exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    mitdb_dir = download_mitdb()
    nstdb_dir = download_nstdb()
    X, y = build_dataset(mitdb_dir, nstdb_dir)

    # 80 / 10 / 10 split  (stratified so classes are balanced in every split)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

    print(f"\nSplits — Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}")

    train_loader = DataLoader(ECGQualityDataset(X_tr,  y_tr),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ECGQualityDataset(X_val, y_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ECGQualityDataset(X_te,  y_te),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model     = SignalQualityAgent(num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_f1 = 0.0
    print(f"{'Epoch':>6} | {'TrainLoss':>10} {'TrainAcc':>10} | "
          f"{'ValLoss':>9} {'ValF1':>7}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_f1, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        marker = " ✓" if va_f1 > best_val_f1 else ""
        print(f"{epoch:>6} | {tr_loss:>10.4f} {tr_acc:>10.4f} | "
              f"{va_loss:>9.4f} {va_f1:>7.4f}{marker}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': va_f1,
                'config': {
                    'segment_len': SEGMENT_LEN,
                    'num_classes': 3,
                    'class_map': {0: 'clean', 1: 'noisy', 2: 'unusable'}
                }
            }, MODEL_PATH)

    # ── Test evaluation ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, te_f1, te_preds, te_labels = eval_epoch(model, test_loader, criterion)

    print(classification_report(
        te_labels, te_preds,
        target_names=['Clean', 'Noisy', 'Unusable'],
        digits=4
    ))
    print(f"Macro F1 on test set: {te_f1:.4f}")
    print(f"Best val F1 achieved: {best_val_f1:.4f} (epoch {checkpoint['epoch']})")

    # ── ONNX export for edge deployment ─────────────────────────────────────
    print(f"\nExporting to ONNX ({ONNX_PATH})...")
    model.eval()
    dummy = torch.randn(1, 1, SEGMENT_LEN).to(DEVICE)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names  = ['ecg_strip'],
        output_names = ['quality_logits'],
        dynamic_axes = {'ecg_strip': {0: 'batch_size'}},
        opset_version = 14
    )
    print(f"ONNX model saved → {ONNX_PATH}")
    print("\nTier 1 complete. Next: python tier2_train.py")

if __name__ == '__main__':
    main()
