# run_independent_lstm.py
"""
Independent Multi-Model Training with UNK handling and score normalization.
Based on the provided baseline code.
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
    def __init__(self, X, yA, yP, yR, L):
        self.X = torch.tensor(X, dtype=torch.long)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.yA[i], self.yP[i], self.yR[i], self.L[i]

# -------------------------
# 特殊組件：自注意力機制層 (用作 Point 模型的最後一拍增強)
# -------------------------
class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)
        
    def forward(self, x, mask=None, causal=True):
        # x: (B, T, dim)
        B, T, C = x.size()
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        # 算出每一拍之間的關聯度 (B, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 1. 處理 Padding Mask (將補零的地方填為極小值)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 2. 【關鍵修改】加入因果遮罩，防止第 t 拍看到 t+1 拍以後的未來資訊
        if causal:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool() # (T, T) 下三角為 True
            scores = scores.masked_fill(~causal_mask.unsqueeze(0), -1e9) # 將右上角（未來）填為 -1e9
            
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return context

# -------------------------
# 獨立模型 1：Action 模型 (深層 Bi-LSTM)
# -------------------------
class ActionModel(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, emb_dim=16, hidden=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature
        ])
        input_dim = self.num_features * emb_dim
        
        # 【關鍵修改】預測下一拍動作同樣強制使用單向 LSTM
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        num_directions = 1
        self.act_head = nn.Linear(hidden * num_directions, n_act)

    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, _ = self.lstm(x)
        o = self.drop(o)
        la = self.act_head(o)
        return la

# -------------------------
# 獨立模型 2：Point 模型 (Bi-LSTM + Attention 機制)
# -------------------------
class PointModel(nn.Module):
    def __init__(self, num_tokens_per_feature, n_pt, emb_dim=16, hidden=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature
        ])
        input_dim = self.num_features * emb_dim
        
        # 【關鍵修改】預測下一拍落點是標準的 Causal 任務，強制使用單向 LSTM，嚴禁雙向偷看
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        
        num_directions = 1
        self.attn = AttentionLayer(hidden * num_directions)
        self.drop = nn.Dropout(dropout)
        self.pt_head = nn.Linear(hidden * num_directions, n_pt)

    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, _ = self.lstm(x)
        
        # Padding Mask (B, 1, T)
        mask = (X[:, :, 0] != PAD_TOKEN).unsqueeze(1)
        
        # 【關鍵修改】啟用因果 Attention 機制
        o_attn = self.attn(o, mask, causal=True)
        o_attn = self.drop(o_attn)
        lp = self.pt_head(o_attn)
        return lp

# -------------------------
# 獨立模型 3：Rally 模型 (標準序列池化分類器)
# -------------------------
class RallyModel(nn.Module):
    def __init__(self, num_tokens_per_feature, emb_dim=16, hidden=128, num_layers=1, dropout=0.2, bidirectional=True):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature
        ])
        input_dim = self.num_features * emb_dim
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        num_directions = 2 if bidirectional else 1
        self.rly_head = nn.Linear(hidden * num_directions, 1)

    def forward(self, X, lengths):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, _ = self.lstm(x)
        o = self.drop(o)
        
        mask = (X[:,:,0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * mask).sum(dim=1) / denom
        
        lr = self.rly_head(mean_hidden).squeeze(1)
        return lr

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
# Main: data prep, independent models, train, inference
# -------------------------
def main(args):
    # =========== Log stdout to file =================
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

    import os
    os.makedirs("result", exist_ok=True)
    log_file = open("result/log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    # ================================================

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

    if len(X_list) == 0:
        raise RuntimeError("No training sequences constructed. Check train data and grouping by rally_uid.")

    global MAXLEN
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

    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2,128), shuffle=False)

    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 初始化三個完全獨立的專門模型
    # ---------------------------------------------------------
    # Action 模型設定為深層 Bi-LSTM (預設 args.layers + 1 增加深度)
    # ---------------------------------------------------------
    # 初始化三個完全獨立的專門模型（修正因果關係版）
    # ---------------------------------------------------------
    model_A = ActionModel(num_tokens_per_feature, n_act, emb_dim=args.emb, hidden=args.hidden,
                          num_layers=args.layers + 1, dropout=args.drop).to(device)
                          
    model_P = PointModel(num_tokens_per_feature, n_pt, emb_dim=args.emb, hidden=args.hidden,
                         num_layers=args.layers, dropout=args.drop).to(device)
                         
    # Rally 模型是預測整局誰得分，屬於全域分類，因此可以保持原有的雙向機制
    model_R = RallyModel(num_tokens_per_feature, emb_dim=args.emb, hidden=args.hidden,
                         num_layers=args.layers, dropout=args.drop, bidirectional=args.bidirectional).to(device)

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1, weight=pt_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()
    
    # 宣告三個獨立模型各自的優化器
    opt_A = torch.optim.Adam(model_A.parameters(), lr=args.lr)
    opt_P = torch.optim.Adam(model_P.parameters(), lr=args.lr)
    opt_R = torch.optim.Adam(model_R.parameters(), lr=args.lr)

    print("--- 開始獨立模型同步訓練 ---")
    for ep in range(1, args.epochs+1):
        model_A.train(); model_P.train(); model_R.train()
        run_loss = 0.0
        for Xb, yAb, yPb, yRb, Lb in train_loader:
            Xb, yAb, yPb, yRb, Lb = Xb.to(device), yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
            
            opt_A.zero_grad(); opt_P.zero_grad(); opt_R.zero_grad()
            
            la = model_A(Xb, Lb)
            lp = model_P(Xb, Lb)
            lr = model_R(Xb, Lb)
            
            loss_A = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
            loss_P = ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
            loss_R = bce_rally(lr, yRb)
            
            loss = 0.375 * loss_A + 0.5 * loss_P + 0.125 * loss_R
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_P.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_R.parameters(), 1.0)
            
            opt_A.step(); opt_P.step(); opt_R.step()
            run_loss += loss.item() * Xb.size(0)

        model_A.eval(); model_P.eval(); model_R.eval()
        val_loss = 0.0
        allA, allAp, allP, allPp, allR, allRp = [], [], [], [], [], []
        with torch.no_grad():
            for Xb, yAb, yPb, yRb, Lb in val_loader:
                Xb, yAb, yPb, yRb, Lb = Xb.to(device), yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
                
                la = model_A(Xb, Lb)
                lp = model_P(Xb, Lb)
                lr = model_R(Xb, Lb)
                
                loss_A = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                loss_P = ce_point(lp.view(-1, lp.size(-1)), yPb.view(-1))
                loss_R = bce_rally(lr, yRb)
                
                loss = 0.4 * loss_A + 0.4 * loss_P + 0.2 * loss_R
                val_loss += loss.item() * Xb.size(0)

                allR += yRb.detach().cpu().tolist(); allRp += torch.sigmoid(lr).detach().cpu().tolist()
                yA_flat = yAb.view(-1).detach().cpu().numpy(); yP_flat = yPb.view(-1).detach().cpu().numpy()
                a_pred = la.argmax(-1).view(-1).detach().cpu().numpy(); p_pred = lp.argmax(-1).view(-1).detach().cpu().numpy()
                
                mA = (yA_flat != -1); mP = (yP_flat != -1)
                allA += yA_flat[mA].tolist(); allAp += a_pred[mA].tolist()
                allP += yP_flat[mP].tolist(); allPp += p_pred[mP].tolist()

        tr_loss = run_loss / len(train_loader.dataset); va_loss = val_loss / len(val_loader.dataset)
        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        except Exception:
            f1A, f1P, auc = 0.0, 0.0, 0.5
        final = 0.4 * f1A + 0.4 * f1P + 0.2 * auc
        print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")

    print("--- 開始對測試集進行獨立預測推論 ---")
    pred_rows = []
    model_A.eval(); model_P.eval(); model_R.eval()
    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            Xg = encode_frame_with_unk(g)
            Xp, T = pad2d_cap(Xg, MAXLEN)
            X_t = torch.tensor(Xp[None, ...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)
            
            # 各自呼叫專屬獨立模型獲取 Logits
            la = model_A(X_t, L_t)
            lp = model_P(X_t, L_t)
            lr = model_R(X_t, L_t)
            
            last_t = L_t.item() - 1
            a_idx = int(torch.argmax(la[0, last_t]).item())
            p_idx = int(torch.argmax(lp[0, last_t]).item())
            # 使用 .item() 獲取標量，徹底杜絕 0維 Tensor 的索引報錯風險
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
    print(f"成功儲存獨立預測提交檔案至: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="inputs/train.csv")
    ap.add_argument("--test", default="inputs/test_new.csv")
    ap.add_argument("--sample", default="result/sample_submission.csv")
    ap.add_argument("--out", default="result/submission.csv")
    ap.add_argument("--epochs", type=int, default=9)
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
    
    MAXLEN = args.max_len
    main(args)