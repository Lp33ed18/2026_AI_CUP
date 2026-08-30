# run_independent_three_lstm.py
"""
Independent Three-LSTM Training script with UNK handling, score normalization, 
Random Truncation, Last-token Validation Alignment, and Best-Weights Checkpoint.
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
# Base FEATURES
# -------------------------
BASE_FEATURES = [
    "sex","handId","strengthId","spinId",
    "pointId","actionId","positionId","strikeId",
    "strikeNumber"
]
PAD_TOKEN = 0

# -------------------------
# Dataset
# -------------------------
class RallyDataset(Dataset):
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

        if self.is_train and L > 1:
            trunc_len = random.randint(1, L.item())
            X[trunc_len:] = PAD_TOKEN
            yA[trunc_len:] = -1
            yP[trunc_len:] = -1
            L = torch.tensor(trunc_len, dtype=torch.long)

        return X, yA, yP, yR, L

# -------------------------
# 💡【修改】拆分為 3 個獨立的 LSTM 模型架構
# -------------------------
class ActionLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, emb_dim=16, hidden=256, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.embs = nn.ModuleList([nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(self.num_features * emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.act_head = nn.Linear(hidden * (2 if bidirectional else 1), n_act)

    def forward(self, X, lengths=None):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, _ = self.lstm(x)
        return self.act_head(self.drop(o))


class PointLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, n_pt, emb_dim=16, hidden=256, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.embs = nn.ModuleList([nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(self.num_features * emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.pt_head = nn.Linear(hidden * (2 if bidirectional else 1), n_pt)

    def forward(self, X, lengths=None):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, _ = self.lstm(x)
        return self.pt_head(self.drop(o))


class RallyLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, emb_dim=16, hidden=256, num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.embs = nn.ModuleList([nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.lstm = nn.LSTM(self.num_features * emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.rly_head = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, X, lengths=None):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, _ = self.lstm(x)
        o = self.drop(o)
        
        mask = (X[:,:,0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * mask).sum(dim=1) / denom
        return self.rly_head(mean_hidden).squeeze(1)

# -------------------------
# Helper functions
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

def normalize_scores(df, cap=11):
    df['scoreSelf'] = pd.to_numeric(df['scoreSelf'], errors='coerce').fillna(-1).astype(int)
    df['scoreOther'] = pd.to_numeric(df['scoreOther'], errors='coerce').fillna(-1).astype(int)
    df['is_deuce'] = ((df['scoreSelf'] >= 10) & (df['scoreOther'] >= 10)).astype(int)
    df['scoreSelf_capped'] = df['scoreSelf'].clip(upper=cap)
    df['scoreOther_capped'] = df['scoreOther'].clip(upper=cap)
    df['score_diff'] = (df['scoreSelf'] - df['scoreOther']).clip(-cap, cap)
    return df

def add_rally_length_features(df, is_inference=False):
    rally_counts = df.groupby("rally_uid").size()
    if is_inference: rally_counts = rally_counts + 1
    df["rally_total_count"] = df["rally_uid"].map(rally_counts)
    df["rally_in_3"] = (df["rally_total_count"] <= 3).astype(int)
    df["rally_4_8"]  = ((df["rally_total_count"] >= 4) & (df["rally_total_count"] <= 8)).astype(int)
    df["rally_long"] = (df["rally_total_count"] >= 9).astype(int)
    df = df.drop(columns=["rally_total_count"])
    return df

def add_lag_features(df):
    grouped = df.groupby("rally_uid")
    df["prev_action"] = grouped["actionId"].shift(1).fillna(0).astype(int)
    df["prev_point"]  = grouped["pointId"].shift(1).fillna(0).astype(int)
    df["prev_pos"]    = grouped["positionId"].shift(1).fillna(0).astype(int)
    df["prev2_action"] = grouped["actionId"].shift(2).fillna(0).astype(int)
    df["prev2_point"]  = grouped["pointId"].shift(2).fillna(0).astype(int)
    df["prev2_pos"]    = grouped["positionId"].shift(2).fillna(0).astype(int)
    
    mask_strike_1 = (df["strikeNumber"] == 1)
    all_lag_cols = ["prev_action", "prev_point", "prev_pos", "prev2_action", "prev2_point", "prev2_pos"]
    df.loc[mask_strike_1, all_lag_cols] = 0
    mask_strike_2 = (df["strikeNumber"] == 2)
    lag2_cols = ["prev2_action", "prev2_point", "prev2_pos"]
    df.loc[mask_strike_2, lag2_cols] = 0
    return df

# -------------------------
# Main
# -------------------------
def main(args):
    class Tee(object):
        def __init__(self, *files): self.files = files
        def write(self, obj):
            for f in self.files: f.write(obj); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    log_file = open("result/log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample) if args.sample else pd.DataFrame()

    train = normalize_scores(train, cap=args.cap)
    test  = normalize_scores(test, cap=args.cap)
    train = add_rally_length_features(train, is_inference=False)
    test  = add_rally_length_features(test, is_inference=True)
    train = add_lag_features(train)
    test  = add_lag_features(test)

    FEATURES = [
        "sex","handId","strengthId","spinId",
        "pointId","actionId","positionId","strikeId",
        "scoreSelf_capped","scoreOther_capped","score_diff","is_deuce","strikeNumber",
        "rally_in_3", "rally_4_8", "rally_long",
        "prev_action", "prev_point", "prev_pos",
        "prev2_action", "prev2_point", "prev2_pos"
    ]

    train["strikeNumber"] = train["strikeNumber"].clip(0, args.max_len-1)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, args.max_len-1)

    cats = {c: pd.Categorical(train[c]).categories for c in FEATURES}

    def encode_frame_with_unk(df):
        outs = []
        for col in FEATURES:
            dtype = pd.CategoricalDtype(categories=cats[col])
            s = pd.Series(df[col])
            codes = s.astype(dtype).cat.codes
            unk_idx = len(cats[col])
            codes = codes.replace(-1, unk_idx)
            codes = codes + 1
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

    if len(X_list) == 0: raise RuntimeError("No training sequences constructed.")

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

    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr, is_train=True)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2,128), shuffle=False)

    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 💡【修改】實例化 3 個完全獨立的模型
    model_act = ActionLSTM(num_tokens_per_feature, n_act, emb_dim=args.emb, hidden=args.hidden, num_layers=max(1,args.layers), dropout=args.drop, bidirectional=args.bidirectional).to(device)
    model_pt  = PointLSTM(num_tokens_per_feature, n_pt, emb_dim=args.emb, hidden=args.hidden, num_layers=max(1,args.layers), dropout=args.drop, bidirectional=args.bidirectional).to(device)
    model_rly = RallyLSTM(num_tokens_per_feature, emb_dim=args.emb, hidden=args.hidden, num_layers=max(1,args.layers), dropout=args.drop, bidirectional=args.bidirectional).to(device)

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()
    
    # 💡【修改】將三個獨立模型的參數包進同一個優化器，進行整合更新
    opt = torch.optim.Adam(
        list(model_act.parameters()) + list(model_pt.parameters()) + list(model_rly.parameters()), 
        lr=args.lr
    )

    best_final_score = -1.0
    best_models_path = "best_independent_three_lstm.pt"

    for ep in range(1, args.epochs+1):
        # 💡【修改】啟用所有模型的訓練模式
        model_act.train(); model_pt.train(); model_rly.train()
        run_loss = 0.0
        for Xb,yAb,yPb,yRb,Lb in train_loader:
            Xb,yAb,yPb,yRb,Lb = Xb.to(device), yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
            opt.zero_grad()
            
            # 💡【修改】分別進行獨立模型的前向傳播
            la = model_act(Xb, Lb)
            lp = model_pt(Xb, Lb)
            lr = model_rly(Xb, Lb)
            
            loss = 0.375*ce_action(la.view(-1,la.size(-1)), yAb.view(-1)) + 0.5*ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + 0.125*bce_rally(lr,yRb)
            loss.backward()
            
            # 💡【修改】對每個模型獨立進行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model_act.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_pt.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_rly.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()*Xb.size(0)

        # 💡【修改】啟用所有模型的評估模式
        model_act.eval(); model_pt.eval(); model_rly.eval()
        val_loss = 0.0
        allA,allAp,allP,allPp,allR,allRp = [],[],[],[],[],[]
        with torch.no_grad():
            for Xb,yAb,yPb,yRb,Lb in val_loader:
                Xb,yAb,yPb,yRb,Lb = Xb.to(device), yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
                la = model_act(Xb, Lb)
                lp = model_pt(Xb, Lb)
                lr = model_rly(Xb, Lb)
                
                loss = 0.4*ce_action(la.view(-1,la.size(-1)), yAb.view(-1)) + 0.4*ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1)) + 0.2*bce_rally(lr,yRb)
                val_loss += loss.item()*Xb.size(0)
                allR += yRb.detach().cpu().tolist(); allRp += torch.sigmoid(lr).detach().cpu().tolist()
                
                yA_np, yP_np = yAb.detach().cpu().numpy(), yPb.detach().cpu().numpy()
                a_pred, p_pred = la.argmax(-1).detach().cpu().numpy(), lp.argmax(-1).detach().cpu().numpy()
                L_np = Lb.detach().cpu().numpy()
                
                for b in range(Xb.size(0)):
                    last_idx = L_np[b] - 1
                    if last_idx >= 0:
                        allA.append(yA_np[b, last_idx]); allAp.append(a_pred[b, last_idx])
                        allP.append(yP_np[b, last_idx]); allPp.append(p_pred[b, last_idx])

        tr_loss = run_loss/len(train_loader.dataset); va_loss = val_loss/len(val_loader.dataset)
        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR))>1 else 0.5
        except Exception:
            f1A,f1P,auc = 0.0,0.0,0.5
        final = 0.4*f1A + 0.4*f1P + 0.2*auc
        print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")

        # 💡【修改】若 Final 創新高，則打包儲存 3 個模型的權重字典
        if final > best_final_score:
            best_final_score = final
            torch.save({
                'action_model': model_act.state_dict(),
                'point_model': model_pt.state_dict(),
                'rally_model': model_rly.state_dict()
            }, best_models_path)

    # 💡【修改】推論前載入 3 個模型表現最好的那組權重
    print(f"\n正在載入最高 Final 分數 ({best_final_score:.4f}) 的最佳獨立模型權重進行推論...")
    checkpoint = torch.load(best_models_path)
    model_act.load_state_dict(checkpoint['action_model'])
    model_pt.load_state_dict(checkpoint['point_model'])
    model_rly.load_state_dict(checkpoint['rally_model'])
    model_act.eval(); model_pt.eval(); model_rly.eval()

    pred_rows = []
    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            Xg = encode_frame_with_unk(g)
            Xp, T = pad2d_cap(Xg, MAXLEN)
            X_t = torch.tensor(Xp[None,...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)
            
            # 💡【修改】利用三個獨立模型各自進行預測
            la = model_act(X_t, L_t)
            lp = model_pt(X_t, L_t)
            lr = model_rly(X_t, L_t)
            
            last_t = L_t.item() - 1
            a_idx = int(torch.argmax(la[0, last_t]).item())
            p_idx = int(torch.argmax(lp[0, last_t]).item())
            s_prob = float(torch.sigmoid(lr).item())
            
            action_pred = int(act_classes[a_idx]) if a_idx < len(act_classes) else int(act_classes[-1])
            point_pred  = int(pt_classes[p_idx])  if p_idx < len(pt_classes)  else int(pt_classes[-1])
            pred_rows.append({"rally_uid": int(rid), "actionId": action_pred, "pointId": point_pred, "serverGetPoint": s_prob})

    pred_df = pd.DataFrame(pred_rows)
    sub_template = sample
    if len(sub_template) > 0 and 'rally_uid' in sub_template.columns:
        out = sub_template.drop(columns=["serverGetPoint","pointId","actionId"], errors="ignore").merge(pred_df, on="rally_uid", how="left")
    else:
        out = pred_df

    cols_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out.reindex(columns=[c for c in cols_order if c in out.columns])
    out.to_csv(args.out, index=False)
    print(f"獨立預測推論完成！預測結果已儲存至: {args.out}")

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
    main(args)