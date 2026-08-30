# run_lstm_final.py
"""
LSTM baseline with UNK handling, score normalization, Random Truncation, and Last-token Validation Alignment.
Save as run_lstm_final.py and run:
python run_lstm_final.py --train train.csv --test test.csv --sample sample_submission.csv --out submission.csv
"""

import argparse
import math
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import sys


# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# -------------------------
# Base FEATURES (score fields will be replaced by derived features)
# -------------------------
BASE_FEATURES = [
    "sex","handId","strengthId","spinId",
    "pointId","actionId","positionId","strikeId",
    # scoreSelf/scoreOther will be replaced by capped versions and derived features
    "strikeNumber"
]
PAD_TOKEN = 0

# -------------------------
# Dataset
# -------------------------
class RallyDataset(Dataset):
    # 💡【修改】引入 is_train 參數來控制是否啟用隨機截斷
    def __init__(self, X, yA, yP, yR, L, is_train=False):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)
        self.is_train = is_train

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, i):
        X = self.X[i].clone()
        yA = self.yA[i].clone()
        yP = self.yP[i].clone()
        yR = self.yR[i]
        L  = self.L[i]

        # 💡【新增】訓練集隨機截斷 (Random Sequence Truncation)
        # 強迫模型在訓練時從隨機時間點（未完結的打球局勢）預測這回合誰贏，防止 AUC 虛高
        if self.is_train and L > 1:
            trunc_len = random.randint(1, L.item())
            
            # 將隨機截斷點之後的特徵全部清空為 PAD_TOKEN (0)
            X[trunc_len:] = PAD_TOKEN
            # 將隨機截斷點之後的 Token 級別標籤設為 -1 (CrossEntropyLoss 會自動 ignore)
            yA[trunc_len:] = -1
            yP[trunc_len:] = -1
            
            # 更新此樣本的有效長度
            L = torch.tensor(trunc_len, dtype=torch.long)

        return X, yA, yP, yR, L

# -------------------------
# LSTM-based MultiTask model
# -------------------------
class MultiTaskLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt, emb_dim=16, hidden=128, num_layers=1, dropout=0.2, bidirectional=True):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.emb_dim = emb_dim
        self.input_dim = self.num_features * emb_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # per-feature embeddings; num_tokens_per_feature already includes UNK
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature
        ])

        self.lstm = nn.LSTM(self.input_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)

        self.drop = nn.Dropout(dropout)
        self.act_head = nn.Linear(hidden * self.num_directions, n_act)
        self.pt_head  = nn.Linear(hidden * self.num_directions, n_pt)
        self.rly_head = nn.Linear(hidden * self.num_directions, 1)

    def forward(self, X, lengths):
        # X: (B, T, F)
        B, T, F = X.size()
        assert F == self.num_features
        es = [emb(X[:,:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)  # (B, T, input_dim)

        o, (hn, cn) = self.lstm(x)  # o: (B, T, hidden * num_directions)
        o = self.drop(o)

        mask = (X[:,:,0] != PAD_TOKEN).float().unsqueeze(-1)  # (B, T, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B,1)
        mean_hidden = (o * mask).sum(dim=1) / denom  # (B, hidden * num_directions)

        la = self.act_head(o)  # token-level action logits (B, T, n_act)
        lp = self.pt_head(o)   # token-level point logits (B, T, n_pt)
        lr = self.rly_head(mean_hidden).squeeze(1)  # rally-level logit (B,)
        return la, lp, lr

# -------------------------
# Helper: padding functions
# -------------------------
def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    out[:len(a)] = a
    return out

def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64)
    out[:len(a)] = a
    return out

def pad2d_cap(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), m)
    out[:T] = a[:T]
    return out, T

# -------------------------
# Score normalization and feature engineering
# -------------------------
def normalize_scores(df, cap=11):
    df['scoreSelf'] = pd.to_numeric(df['scoreSelf'], errors='coerce').fillna(-1).astype(int)
    df['scoreOther'] = pd.to_numeric(df['scoreOther'], errors='coerce').fillna(-1).astype(int)
    df['is_deuce'] = ((df['scoreSelf'] >= 10) & (df['scoreOther'] >= 10)).astype(int)
    df['scoreSelf_capped'] = df['scoreSelf'].clip(upper=cap)
    df['scoreOther_capped'] = df['scoreOther'].clip(upper=cap)
    df['score_diff'] = (df['scoreSelf'] - df['scoreOther']).clip(-cap, cap)
    return df

# -------------------------
# Main: data prep, model, train, inference
# -------------------------
def main(args):
    # ===========  ==================================
    # Custom Tee to log stdout to file
    class Tee(object):
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open("result/log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    # ===========  ==================================

    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample) if args.sample else pd.DataFrame()

    train = normalize_scores(train, cap=args.cap)
    test  = normalize_scores(test, cap=args.cap)

    FEATURES = [
        "sex","handId","strengthId","spinId",
        "pointId","actionId","positionId","strikeId",
        "scoreSelf_capped","scoreOther_capped","score_diff","is_deuce","strikeNumber"
    ]

    train["strikeNumber"] = train["strikeNumber"].clip(0, args.max_len-1)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, args.max_len-1)

    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}

    def encode_frame_with_unk(df):
        outs = []
        for col in FEATURES:
            dtype = pd.CategoricalDtype(categories=cats[col])
            s = pd.Series(df[col])
            codes = s.astype(dtype).cat.codes  # unseen -> -1
            unk_idx = len(cats[col])
            codes = codes.replace(-1, unk_idx)
            codes = codes + 1  # shift so 0 reserved for PAD
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)

    X_list, yA_list, yP_list, yR_list, L_list = [], [], [], [], []
    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2: continue
        X = encode_frame_with_unk(g)[:-1]
        yA = g["actionId"].values[1:].astype(np.int64)
        yP = g["pointId"].values[1:].astype(np.int64)
        X_list.append(X); yA_list.append(yA); yP_list.append(yP)
        yR_list.append(int(g["serverGetPoint"].iloc[0] if "serverGetPoint" in g.columns else 0)); L_list.append(len(X))

    if len(X_list) == 0:
        raise RuntimeError("No training sequences constructed. Check train data and grouping by rally_uid.")

    MAXLEN = min(max(L_list), args.max_len)

    X_all  = np.stack([pad2d(s, MAXLEN) for s in X_list])
    yA_all = np.stack([pad1d(s, MAXLEN) for s in yA_list])
    yP_all = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all = np.array(yR_list, dtype=np.float32)
    L_all  = np.array(L_list, dtype=np.int64)

    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes)
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes)
    act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}
    yA_all = np.vectorize(lambda v: act_id2idx.get(v, -1))(yA_all)
    yP_all = np.vectorize(lambda v: pt_id2idx.get(v, -1))(yP_all)

    idx = np.arange(len(X_all))
    tr_idx, va_idx = train_test_split(idx, test_size=args.val_size, random_state=SEED, stratify=(yR_all>0.5))
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    yA_tr, yA_va = yA_all[tr_idx], yA_all[va_idx]
    yP_tr, yP_va = yP_all[tr_idx], yP_all[va_idx]
    yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]
    L_tr,  L_va  = L_all[tr_idx],  L_all[va_idx]

    act_counts = np.bincount(yA_tr[yA_tr!=-1].ravel(), minlength=n_act) + 1
    pt_counts  = np.bincount(yP_tr[yP_tr!=-1].ravel(), minlength=n_pt) + 1
    act_w = torch.tensor(1.0/act_counts, dtype=torch.float32); act_w = (act_w * (n_act/act_w.sum()))
    pt_w  = torch.tensor(1.0/pt_counts,  dtype=torch.float32); pt_w  = (pt_w  * (n_pt /pt_w.sum()))

    # 💡【修改】顯式指定 train_ds 使用隨機截斷 (is_train=True)，而 val_ds 保持完整不截斷 (is_train=False)
    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr, is_train=True)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2,128), shuffle=False)

    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskLSTM(num_tokens_per_feature, n_act, n_pt,
                         emb_dim=args.emb, hidden=args.hidden,
                         num_layers=max(1,args.layers), dropout=args.drop,
                         bidirectional=args.bidirectional).to(device)

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs+1):
        model.train(); run_loss=0.0
        for Xb,yAb,yPb,yRb,Lb in train_loader:
            Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
            opt.zero_grad()
            la,lp,lr = model(Xb,Lb)
            loss = 0.375*ce_action(la.view(-1,la.size(-1)), yAb.view(-1)) + 0.5*ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + 0.125*bce_rally(lr,yRb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            run_loss += loss.item()*Xb.size(0)

        model.eval(); val_loss=0.0
        allA,allAp,allP,allPp,allR,allRp = [],[],[],[],[],[]
        with torch.no_grad():
            for Xb,yAb,yPb,yRb,Lb in val_loader:
                Xb,yAb,yPb,yRb,Lb = Xb.to(device),yAb.to(device),yPb.to(device),yRb.to(device),Lb.to(device)
                la,lp,lr = model(Xb,Lb)
                loss = 0.4*ce_action(la.view(-1,la.size(-1)), yAb.view(-1)) + 0.4*ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + 0.2*bce_rally(lr,yRb)
                val_loss += loss.item()*Xb.size(0)

                allR += yRb.detach().cpu().tolist(); allRp += torch.sigmoid(lr).detach().cpu().tolist()
                
                # 💡【修改】驗證集評估邏輯對齊 LB 計分標準：僅抽取每一局的「最後一個有效拍」來計算 F1
                yA_np = yAb.detach().cpu().numpy()
                yP_np = yPb.detach().cpu().numpy()
                a_pred = la.argmax(-1).detach().cpu().numpy()
                p_pred = lp.argmax(-1).detach().cpu().numpy()
                L_np = Lb.detach().cpu().numpy()
                
                for b in range(Xb.size(0)):
                    last_idx = L_np[b] - 1  # 找到未補零前的最後一個有效位置
                    if last_idx >= 0:
                        allA.append(yA_np[b, last_idx])
                        allAp.append(a_pred[b, last_idx])
                        allP.append(yP_np[b, last_idx])
                        allPp.append(p_pred[b, last_idx])

        tr_loss = run_loss/len(train_loader.dataset); va_loss = val_loss/len(val_loader.dataset)
        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR))>1 else 0.5
        except Exception:
            f1A,f1P,auc = 0.0,0.0,0.5
        final = 0.4*f1A + 0.4*f1P + 0.2*auc
        print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")


    pred_rows = []
    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            Xg = encode_frame_with_unk(g)
            Xp, T = pad2d_cap(Xg, MAXLEN)
            X_t = torch.tensor(Xp[None,...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)
            la,lp,lr = model(X_t, L_t)
            last_t = L_t.item() - 1
            a_idx = int(torch.argmax(la[0, last_t]).item())
            p_idx = int(torch.argmax(lp[0, last_t]).item())
            s_prob = float(torch.sigmoid(lr).item())
            action_pred = int(act_classes[a_idx]) if a_idx < len(act_classes) else int(act_classes[-1])
            point_pred  = int(pt_classes[p_idx])  if p_idx < len(pt_classes)  else int(pt_classes[-1])
            # append in requested order: rally_uid, actionId, pointId, serverGetPoint
            pred_rows.append({"rally_uid": int(rid), "actionId": action_pred, "pointId": point_pred, "serverGetPoint": s_prob})

    pred_df = pd.DataFrame(pred_rows)

    sub_template = sample
    if len(sub_template) > 0 and 'rally_uid' in sub_template.columns:
        out = sub_template.drop(columns=["serverGetPoint","pointId","actionId"], errors="ignore").merge(pred_df, on="rally_uid", how="left")
    else:
        out = pred_df

    # ensure column order as requested
    cols_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out.reindex(columns=[c for c in cols_order if c in out.columns])

    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="inputs/train.csv")
    ap.add_argument("--test", default="inputs/test_new.csv")
    ap.add_argument("--sample", default="result/sample_submission.csv")
    ap.add_argument("--out", default="result/submission.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--cap", type=int, default=11)
    ap.add_argument("--bidirectional", type=bool, default=True)
    args = ap.parse_args()
    # MAXLEN must be defined before inference; compute a safe default if not set by training
    # We'll set a global MAXLEN based on args.max_len; training will override it
    MAXLEN = args.max_len
    main(args)