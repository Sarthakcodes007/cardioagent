"""
CardioAgent — Tier 2 v4: 12-Lead with Lead Dropout Augmentation

Key fix from v3:
  Single-lead AUROC was 0.52 (catastrophic) because model only saw
  12-lead data during training. When deployed on wearable (Lead-I only),
  it collapsed completely.

  Fix: Lead Dropout Augmentation during training:
    - 30% of batches: zero out all leads except Lead I (simulate wearable)
    - 20% of batches: randomly drop 1-5 leads (simulate partial contact)
    - 50% of batches: full 12-lead (hospital quality)

  This forces the model to learn lead-invariant features that work on
  both hospital 12-lead ECGs and single-lead wearable devices.

Expected results:
  12-lead AUROC : 0.87-0.90 (slight drop from augmentation, acceptable)
  Single-lead AUROC: 0.78-0.84 (massive improvement from 0.52)

Run: python tier2_12lead.py
"""

import os, math, warnings, random, ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, classification_report, f1_score
import wfdb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── CONFIG ───────────────────────────────────────────────────────────────────
N_LEADS      = 12
SEGMENT_LEN  = 1000
PATCH_SIZE   = 5
N_PATCHES    = SEGMENT_LEN // PATCH_SIZE
PATCH_DIM    = N_LEADS * PATCH_SIZE         # 60

D_MODEL      = 256
N_HEADS      = 8
N_LAYERS     = 6
D_FF         = 512
DROPOUT      = 0.1
N_CLASSES    = 5
BATCH_SIZE   = 64
EPOCHS       = 70
LR           = 3e-4
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA  = 2.0

# Lead dropout probabilities
P_SINGLE_LEAD  = 0.30   # simulate wearable: keep only Lead I
P_RANDOM_DROP  = 0.20   # randomly drop 1-5 leads

DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR   = '/workspace/cardioagent/data/ptbxl'
MODEL_PATH = '/workspace/cardioagent/models/tier2_12lead.pt'
ONNX_PATH  = '/workspace/cardioagent/models/tier2_12lead.onnx'
SEED       = 42
CLASS_NAMES = ['Normal', 'AFib', 'STEMI/MI', 'BBB', 'Other']

MI_CODES  = {'ASMI','IMI','ILMI','ALMI','AMI','LMI','PMI','IPLMI','IPMI',
             'INJAL','INJAS','INJIL','INJIN','INJLA',
             'ISCAL','ISCAN','ISCIN','ISCLA','ISCIL','ISCAS','ISC_'}
BBB_CODES = {'IRBBB','CRBBB','CLBBB','LBBB','RBBB','IVCD','LAFB','LPFB'}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── LABEL MAPPING ─────────────────────────────────────────────────────────────
def assign_superclass(scp):
    if 'AFIB' in scp or 'AFLT' in scp: return 1
    codes = {k for k,v in scp.items() if v >= 50}
    if codes & MI_CODES:   return 2
    if codes & BBB_CODES:  return 3
    if codes & {'NORM'}:   return 0
    return 4

# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_metadata():
    df = pd.read_csv(os.path.join(DATA_DIR,'ptbxl_database.csv'), index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)
    df['label']     = df['scp_codes'].apply(assign_superclass)
    print(f"PTB-XL: {len(df)} records")
    for i, name in enumerate(CLASS_NAMES):
        n = (df['label']==i).sum()
        print(f"  {i}={name}: {n:,} ({100*n/len(df):.1f}%)")
    return df

def load_12lead(df, desc="Loading"):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {desc}"):
        try:
            rec = wfdb.rdrecord(os.path.join(DATA_DIR, row['filename_lr']))
            sig = rec.p_signal.astype(np.float32)        # (1000, 12)
            if sig.shape != (SEGMENT_LEN, N_LEADS): continue
            if np.isnan(sig).any(): continue
            sig = sig.T                                   # (12, 1000)
            for i in range(N_LEADS):
                sig[i] = (sig[i]-sig[i].mean())/(sig[i].std()+1e-8)
            X.append(sig); y.append(int(row['label']))
        except: continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def load_single_lead(df, desc="Loading 1L"):
    """Single Lead-I padded to 12-lead format for evaluation."""
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {desc}"):
        try:
            rec = wfdb.rdrecord(os.path.join(DATA_DIR, row['filename_lr']))
            sig = rec.p_signal[:,0].astype(np.float32)   # Lead I only
            if len(sig) != SEGMENT_LEN or np.isnan(sig).any(): continue
            sig = (sig-sig.mean())/(sig.std()+1e-8)
            sig_12 = np.zeros((N_LEADS, SEGMENT_LEN), dtype=np.float32)
            sig_12[0] = sig
            X.append(sig_12); y.append(int(row['label']))
        except: continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def split(df):
    return df[df['strat_fold']<=8], df[df['strat_fold']==9], df[df['strat_fold']==10]

# ── DATASET ───────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def oversampled_loader(X, y, bs):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    w      = np.minimum(counts.max()/(counts+1e-6), 5.0)
    sw     = torch.FloatTensor([w[l] for l in y])
    samp   = WeightedRandomSampler(sw, len(sw), replacement=True)
    return DataLoader(ECGDataset(X,y), bs, sampler=samp,
                      num_workers=4, pin_memory=True)

# ── LEAD DROPOUT AUGMENTATION ─────────────────────────────────────────────────
def apply_lead_dropout(X: torch.Tensor) -> torch.Tensor:
    """
    Applied per-batch during training.
    X shape: (B, 12, 1000)

    30% chance → single Lead-I only  (wearable simulation)
    20% chance → drop 1-5 random leads  (partial contact)
    50% chance → full 12-lead  (hospital quality, no change)
    """
    X = X.clone()
    B = X.size(0)
    r = torch.rand(B)

    for i in range(B):
        if r[i] < P_SINGLE_LEAD:
            # Keep only Lead I (index 0)
            lead_i = X[i, 0].clone()
            X[i]   = 0.0
            X[i, 0] = lead_i

        elif r[i] < P_SINGLE_LEAD + P_RANDOM_DROP:
            # Drop 1-5 random leads (not Lead I)
            n_drop = random.randint(1, 5)
            drops  = random.sample(range(1, N_LEADS), n_drop)
            X[i, drops] = 0.0

    return X

# ── MODEL ─────────────────────────────────────────────────────────────────────
class MultiLeadPatchEmbedding(nn.Module):
    def __init__(self, n_leads, patch_size, d_model):
        super().__init__()
        self.n_leads = n_leads; self.ps = patch_size
        self.proj = nn.Linear(n_leads*patch_size, d_model)

    def forward(self, x):                       # (B, 12, 1000)
        B, L, S = x.shape
        n = S // self.ps
        x = x.reshape(B, L, n, self.ps)         # (B, 12, 200, 5)
        x = x.permute(0, 2, 1, 3)               # (B, 200, 12, 5)
        x = x.reshape(B, n, L*self.ps)          # (B, 200, 60)
        return self.proj(x)                      # (B, 200, D)

class SinPE(nn.Module):
    def __init__(self, d, ml=300, dr=0.1):
        super().__init__(); self.drop=nn.Dropout(dr)
        pe=torch.zeros(ml,d); pos=torch.arange(0,ml).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*-(math.log(10000)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x): return self.drop(x+self.pe[:,:x.size(1)])

class ArrhythmiaClassifier12Lead(nn.Module):
    """
    12-Lead ECG Transformer with lead dropout training.
    Input  : (B, 12, 1000)
    Output : (B, 5)
    Works at inference with any number of leads including single Lead-I.
    """
    def __init__(self):
        super().__init__()
        self.patch_embed = MultiLeadPatchEmbedding(N_LEADS, PATCH_SIZE, D_MODEL)
        self.pos_enc     = SinPE(D_MODEL, N_PATCHES+1, DROPOUT)
        self.cls_token   = nn.Parameter(torch.randn(1,1,D_MODEL)*0.02)
        enc = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT,
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, N_LAYERS)
        self.norm        = nn.LayerNorm(D_MODEL)
        self.drop        = nn.Dropout(DROPOUT)
        self.head        = nn.Linear(D_MODEL, N_CLASSES)
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)

    def forward(self, x):
        B   = x.size(0)
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B,-1,-1)
        x   = torch.cat([cls,x],dim=1)
        x   = self.pos_enc(x)
        x   = self.transformer(x)
        return self.head(self.drop(self.norm(x[:,0])))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            p = torch.softmax(self.forward(x),-1)
        return p.argmax(-1), p.max(-1).values, p

# ── LOSS ──────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma=gamma
    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, reduction='none')
        pt  = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

# ── TRAIN / EVAL ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, scheduler):
    model.train()
    tot, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        X    = apply_lead_dropout(X)            # ← augmentation here
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out  = model(X); loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        tot += loss.item()*len(y)
        correct += (out.argmax(1)==y).sum().item(); total += len(y)
    return tot/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    tot=0.0; preds,labels,probs=[],[],[]
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE); out=model(X)
        tot += criterion(out,y).item()*len(y)
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        probs.extend(torch.softmax(out,-1).cpu().tolist())
    f1 = f1_score(labels,preds,average='macro',zero_division=0)
    try:    auroc=roc_auc_score(labels,probs,multi_class='ovr',average='macro')
    except: auroc=0.0
    return tot/len(loader.dataset), f1, auroc, preds, labels, probs

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*65}")
    print("CardioAgent — Tier 2 v4: 12-Lead + Lead Dropout Augmentation")
    print(f"Device: {DEVICE} | Lead dropout: {P_SINGLE_LEAD*100:.0f}% single, "
          f"{P_RANDOM_DROP*100:.0f}% random drop")
    print(f"{'='*65}\n")

    os.makedirs('/workspace/cardioagent/models', exist_ok=True)

    df                    = load_metadata()
    df_tr, df_va, df_te   = split(df)
    print(f"\nSplits — Train:{len(df_tr):,}  Val:{len(df_va):,}  Test:{len(df_te):,}")

    print("\nLoading 12-lead waveforms...")
    X_tr,y_tr = load_12lead(df_tr,"Train")
    X_va,y_va = load_12lead(df_va,"Val  ")
    X_te,y_te = load_12lead(df_te,"Test ")

    counts = np.bincount(y_tr, minlength=N_CLASSES)
    print(f"Train counts: {dict(zip(CLASS_NAMES,counts))}")

    train_loader = oversampled_loader(X_tr,y_tr,BATCH_SIZE)
    val_loader   = DataLoader(ECGDataset(X_va,y_va),BATCH_SIZE,False,
                               num_workers=4,pin_memory=True)
    test_loader  = DataLoader(ECGDataset(X_te,y_te),BATCH_SIZE,False,
                               num_workers=4,pin_memory=True)

    model     = ArrhythmiaClassifier12Lead().to(DEVICE)
    criterion = FocalLoss(gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'Ep':>4} | {'TrLoss':>8} {'TrAcc':>7} | {'VaF1':>6} {'VaAUROC':>8}")
    print("─"*50)

    best_auroc = 0.0
    for epoch in range(1, EPOCHS+1):
        tl,ta                        = train_epoch(model,train_loader,
                                                    optimizer,criterion,
                                                    scaler,scheduler)
        _,vf,va,_,_,_                = eval_epoch(model,val_loader,criterion)
        marker = " ✓" if va>best_auroc else ""
        print(f"{epoch:>4} | {tl:>8.4f} {ta:>7.4f} | {vf:>6.4f} {va:>8.4f}{marker}")
        if va>best_auroc:
            best_auroc=va
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
                        'val_auroc':va,
                        'config':{'n_leads':N_LEADS,'segment_len':SEGMENT_LEN,
                                  'patch_size':PATCH_SIZE,'class_names':CLASS_NAMES}},
                       MODEL_PATH)

    # ── 12-lead test ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nTest — 12-lead (hospital ECG)\n{'='*65}")
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
    _,tf,ta,tp,tl,_ = eval_epoch(model,test_loader,criterion)
    print(classification_report(tl,tp,target_names=CLASS_NAMES,digits=4))
    print(f"Macro F1: {tf:.4f}  |  Macro AUROC: {ta:.4f}")
    auroc_12l = ta

    # ── Single Lead-I test ───────────────────────────────────────────────────
    print(f"\n{'='*65}\nTest — Single Lead-I (wearable deployment)\n{'='*65}")
    X_te_1l,y_te_1l = load_single_lead(df_te,"Test 1L")
    test_1l = DataLoader(ECGDataset(X_te_1l,y_te_1l),BATCH_SIZE,False,
                          num_workers=4)
    _,tf_1l,ta_1l,tp_1l,_,_ = eval_epoch(model,test_1l,criterion)
    print(classification_report(tl,tp_1l,target_names=CLASS_NAMES,digits=4))
    print(f"Macro F1: {tf_1l:.4f}  |  Macro AUROC: {ta_1l:.4f}")

    print(f"\n{'─'*50}")
    print(f"12-lead AUROC   (hospital)  : {auroc_12l:.4f}")
    print(f"Single-lead AUROC (wearable): {ta_1l:.4f}")
    print(f"Lead reduction gap          : {auroc_12l-ta_1l:.4f}")
    print(f"Best val AUROC: {best_auroc:.4f}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    print(f"\nExporting ONNX → {ONNX_PATH}")
    model.eval()
    dummy = torch.randn(1,N_LEADS,SEGMENT_LEN).to(DEVICE)
    torch.onnx.export(model, dummy, ONNX_PATH,
                      input_names=['ecg_12lead'],output_names=['class_logits'],
                      dynamic_axes={'ecg_12lead':{0:'batch'}},opset_version=14)
    print("Tier 2 v4 complete.")

if __name__ == '__main__':
    main()
