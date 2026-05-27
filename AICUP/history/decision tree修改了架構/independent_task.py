# run_independent_lstm_xgboost.py
"""
Independent Multi-Model Training:
- ActionModel & RallyModel: Deep Learning (PyTorch LSTM)
- PointModel: Machine Learning (XGBoost with Tabular Lag Features + LSTM deep features)
This version integrates LSTM hidden vectors into XGBoost (LSTM Hidden -> XGBoost ensemble).
Notes:
- This is the "Version A (strongest)" default: action_hidden_dim=256, rally_hidden_dim=128 (total 384 dims).
- Extraction of per-timestep hidden vectors can be memory/time heavy. Adjust --action_feat_dim and --rally_feat_dim if needed.
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
import xgboost as xgb  # 引入 XGBoost

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
# 獨立模型 1：Action 模型 (深層單向 LSTM)
# -------------------------
class ActionModel(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, emb_dim=16, hidden=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature
        ])
        input_dim = self.num_features * emb_dim
        
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=False)
        self.drop = nn.Dropout(dropout)
        num_directions = 1
        self.act_head = nn.Linear(hidden * num_directions, n_act)

    def forward(self, X, lengths, return_hidden=False):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, (h, c) = self.lstm(x)
        o = self.drop(o)
        la = self.act_head(o)
        if return_hidden:
            # h shape: (num_layers, batch, hidden)
            last_h = h[-1]  # (batch, hidden)
            return la, last_h
        return la

# -------------------------
# 獨立模型 2：Rally 模型 (全域雙向 LSTM 池化分類器)
# -------------------------
class RallyModel(nn.Module):
    def __init__(self, num_tokens_per_feature, emb_dim=16, hidden=128, num_layers=1, dropout=0.3, bidirectional=False):
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

    def forward(self, X, lengths, return_hidden=False):
        es = [emb(X[:,:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(es, dim=-1)
        o, (h, c) = self.lstm(x)
        o = self.drop(o)
        
        mask = (X[:,:,0] != PAD_TOKEN).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_hidden = (o * mask).sum(dim=1) / denom
        
        lr = self.rly_head(mean_hidden).squeeze(1)
        if return_hidden:
            return lr, mean_hidden
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
# Utility: extract per-timestep LSTM hidden vectors (full vectors) for each row
# Returns two arrays: action_hidden_matrix (N_rows x action_feat_dim), rally_hidden_matrix (N_rows x rally_feat_dim)
# -------------------------
def extract_lstm_hidden_vectors(df, model_A, model_R, cats, FEATURES, MAXLEN, device, action_feat_dim, rally_feat_dim):
    """
    For each rally_uid and each row (stroke), run the LSTM on sequence up to that stroke (inclusive)
    and extract:
      - action_hidden_vector: last-layer hidden vector (trim/pad to action_feat_dim)
      - rally_hidden_vector: pooled hidden vector (trim/pad to rally_feat_dim)
    Returns two lists aligned with df rows order.
    """
    model_A.eval(); model_R.eval()
    action_vecs = []
    rally_vecs = []
    with torch.no_grad():
        for rid, g in df.groupby("rally_uid"):
            # encode full rally once
            outs = []
            for col in FEATURES:
                dtype = pd.CategoricalDtype(categories=cats[col])
                s = pd.Series(g[col])
                codes = s.astype(dtype).cat.codes
                unk_idx = len(cats[col])
                codes = codes.replace(-1, unk_idx)
                codes = codes + 1
                outs.append(np.asarray(codes, dtype=np.int64))
            Xg_full = np.stack(outs, axis=1)  # shape (T, feat)
            T_full = len(Xg_full)
            # for each stroke i (0..T_full-1), run up to i (inclusive) to get hidden representing current context
            for i in range(T_full):
                seq = Xg_full[:i+1]
                Xp, T = pad2d_cap(seq, MAXLEN)
                X_t = torch.tensor(Xp[None, ...], dtype=torch.long, device=device)
                L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)
                # get hidden
                _, hA = model_A(X_t, L_t, return_hidden=True)   # (batch, hiddenA)
                _, hR = model_R(X_t, L_t, return_hidden=True)   # (batch, hiddenR)
                hA = hA[0].cpu().numpy()
                hR = hR[0].cpu().numpy()
                # trim or pad to desired dims
                if len(hA) >= action_feat_dim:
                    a_vec = hA[:action_feat_dim]
                else:
                    a_vec = np.concatenate([hA, np.zeros(action_feat_dim - len(hA), dtype=np.float32)])
                if len(hR) >= rally_feat_dim:
                    r_vec = hR[:rally_feat_dim]
                else:
                    r_vec = np.concatenate([hR, np.zeros(rally_feat_dim - len(hR), dtype=np.float32)])
                action_vecs.append(a_vec.astype(np.float32))
                rally_vecs.append(r_vec.astype(np.float32))
    action_matrix = np.vstack(action_vecs) if len(action_vecs) else np.zeros((0, action_feat_dim), dtype=np.float32)
    rally_matrix = np.vstack(rally_vecs) if len(rally_vecs) else np.zeros((0, rally_feat_dim), dtype=np.float32)
    return action_matrix, rally_matrix

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
    rids_list = []  # 用來記錄排序好的每一局 ID，方便後續對齊 XGBoost
    
    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2: continue
        X = encode_frame_with_unk(g)[:-1]
        yA = g["actionId"].values[1:].astype(np.int64)
        yP = g["pointId"].values[1:].astype(np.int64)
        X_list.append(X); yA_list.append(yA); yP_list.append(yP)
        yR_list.append(int(g["serverGetPoint"].iloc[0] if "serverGetPoint" in g.columns else 0)); L_list.append(len(X))
        rids_list.append(rid)

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

    # 獲取訓練集與驗證集對應的 rally_uid 集合
    tr_rids = set([rids_list[i] for i in tr_idx])
    va_rids = set([rids_list[i] for i in va_idx])

    act_counts = np.bincount(yA_tr[yA_tr!=-1].ravel(), minlength=n_act) + 1
    act_w = torch.tensor(1.0/act_counts, dtype=torch.float32); act_w = (act_w * (n_act/act_w.sum()))

    train_ds = RallyDataset(X_tr, yA_tr, yP_tr, yR_tr, L_tr)
    val_ds   = RallyDataset(X_va, yA_va, yP_va, yR_va, L_va)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2,128), shuffle=False)

    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    # 初始化 PyTorch 模型（提前初始化以便抽取 hidden features）
    # ---------------------------------------------------------
    model_A = ActionModel(num_tokens_per_feature, n_act, emb_dim=args.emb, hidden=args.hidden,
                          num_layers=args.layers + 1, dropout=args.drop).to(device)
                         
    model_R = RallyModel(num_tokens_per_feature, emb_dim=args.emb, hidden=args.hidden_rally,
                         num_layers=args.layers, dropout=args.drop, bidirectional=args.bidirectional).to(device)

    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()
    
    opt_A = torch.optim.Adam(model_A.parameters(), lr=args.lr)
    opt_R = torch.optim.Adam(model_R.parameters(), lr=args.lr)

    # Optional pretraining of LSTM models so extracted hidden features are meaningful.
    pretrain_epochs = args.pretrain_epochs
    if pretrain_epochs > 0:
        print(f"--- Pretraining Action & Rally for {pretrain_epochs} epochs to produce meaningful hidden features ---")
        for ep in range(1, pretrain_epochs+1):
            model_A.train(); model_R.train()
            run_loss = 0.0
            for Xb, yAb, yPb, yRb, Lb in train_loader:
                Xb, yAb, yPb, yRb, Lb = Xb.to(device), yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
                opt_A.zero_grad(); opt_R.zero_grad()
                la = model_A(Xb, Lb)
                lr = model_R(Xb, Lb)
                loss_A = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                loss_R = bce_rally(lr, yRb)
                loss = 0.75 * loss_A + 0.25 * loss_R
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model_R.parameters(), 1.0)
                opt_A.step(); opt_R.step()
                run_loss += loss.item() * Xb.size(0)
            tr_loss = run_loss / len(train_loader.dataset)
            print(f"[Pretrain Epoch {ep}/{pretrain_epochs}] train_loss={tr_loss:.4f}")
        print("--- Pretraining done ---")

    # ---------------------------------------------------------
    # 【時序特徵工程】建構專供 XGBoost 預測 pointId 的表格特徵
    # 此處我們會加入 LSTM hidden vectors (action_h_*, rally_h_*)
    # ---------------------------------------------------------
    print("--- 正在為 XGBoost 生成時序特徵 (Lag Features + LSTM hidden vectors) ---")
    from sklearn.utils.class_weight import compute_sample_weight
    
    train_xgb_base = train.copy()
    
    # 1. 對基礎特徵進行與 PyTorch 邏輯一致的 Label Encoding，確保無字串型態干擾
    for col in FEATURES:
        dtype = pd.CategoricalDtype(categories=cats[col])
        train_xgb_base[f"{col}_enc"] = train_xgb_base[col].astype(dtype).cat.codes + 1
        
    xgb_base_features = [f"{col}_enc" for col in FEATURES]
    
    # 2. 建立 Lag 1 與 Lag 2，移除 .fillna(0)，保留 NaN 讓 XGBoost 原生處理
    for col in FEATURES:
        train_xgb_base[f"{col}_enc_lag1"] = train_xgb_base.groupby("rally_uid")[f"{col}_enc"].shift(1)
        train_xgb_base[f"{col}_enc_lag2"] = train_xgb_base.groupby("rally_uid")[f"{col}_enc"].shift(2)
    
    # 目標特徵：下一拍的 pointId
    train_xgb_base["target_pointId"] = train_xgb_base.groupby("rally_uid")["pointId"].shift(-1)
    train_xgb_base = train_xgb_base.dropna(subset=["target_pointId"]).copy()
    train_xgb_base["target_mapped"] = train_xgb_base["target_pointId"].map(pt_id2idx)
    
    # 確保目標值沒有缺失值且型態正確
    train_xgb_base = train_xgb_base.dropna(subset=["target_mapped"]).copy()
    train_xgb_base["target_mapped"] = train_xgb_base["target_mapped"].astype(int)

    # 2.5 Extract LSTM hidden vectors aligned with train rows
    print("Extracting LSTM hidden vectors for training set (this may take time and memory)...")
    action_matrix, rally_matrix = extract_lstm_hidden_vectors(
        train, model_A, model_R, cats, FEATURES, MAXLEN, device,
        action_feat_dim=args.action_feat_dim, rally_feat_dim=args.rally_feat_dim
    )
    # Build keys to align rows: rally_uid + strikeNumber
    train_loc = train.copy().reset_index(drop=True)
    train_loc["key"] = train_loc["rally_uid"].astype(str) + "_" + train_loc["strikeNumber"].astype(str)
    # train_xgb_base corresponds to train rows except last per rally (we dropped last stroke)
    train_xgb_base["key"] = train_xgb_base["rally_uid"].astype(str) + "_" + train_xgb_base["strikeNumber"].astype(str)
    # action_matrix and rally_matrix are in the same order as iterating train.groupby("rally_uid") and per-timestep
    if action_matrix.shape[0] != len(train_loc):
        print("Warning: extracted hidden vectors length mismatch. Filling zeros for hidden features.")
        for i in range(args.action_feat_dim):
            train_xgb_base[f"action_h_{i}"] = 0.0
        for j in range(args.rally_feat_dim):
            train_xgb_base[f"rally_h_{j}"] = 0.0
    else:
        # attach vectors to train_loc
        for i in range(args.action_feat_dim):
            train_loc[f"action_h_{i}"] = action_matrix[:, i]
        for j in range(args.rally_feat_dim):
            train_loc[f"rally_h_{j}"] = rally_matrix[:, j]
        # merge into train_xgb_base by key
        cols_to_merge = ["key"] + [f"action_h_{i}" for i in range(args.action_feat_dim)] + [f"rally_h_{j}" for j in range(args.rally_feat_dim)]
        train_xgb_base = train_xgb_base.merge(train_loc[cols_to_merge], on="key", how="left")
        # fillna with 0
        for i in range(args.action_feat_dim):
            train_xgb_base[f"action_h_{i}"] = train_xgb_base[f"action_h_{i}"].fillna(0.0)
        for j in range(args.rally_feat_dim):
            train_xgb_base[f"rally_h_{j}"] = train_xgb_base[f"rally_h_{j}"].fillna(0.0)
        train_xgb_base = train_xgb_base.drop(columns=["key"])

    # 整合所有為 XGBoost 設計的編碼與時序特徵 + LSTM hidden vectors
    xgb_features = xgb_base_features + [f"{col}_enc_lag1" for col in FEATURES] + [f"{col}_enc_lag2" for col in FEATURES]
    xgb_features += [f"action_h_{i}" for i in range(args.action_feat_dim)] + [f"rally_h_{j}" for j in range(args.rally_feat_dim)]

    # 切分出與 PyTorch 完全對齊的 XGBoost 訓練與驗證集
    train_xgb_df = train_xgb_base[train_xgb_base["rally_uid"].isin(tr_rids)]
    val_xgb_df   = train_xgb_base[train_xgb_base["rally_uid"].isin(va_rids)]

    X_train_xgb = train_xgb_df[xgb_features].astype(float)
    y_train_xgb = train_xgb_df["target_mapped"]
    X_val_xgb   = val_xgb_df[xgb_features].astype(float)
    y_val_xgb   = val_xgb_df["target_mapped"]

    print(f"XGBoost 訓練集樣本數: {len(X_train_xgb)}, 驗證集樣本數: {len(X_val_xgb)}")
    print("--- 開始訓練 pointId 的 XGBoost 模型 ---")
    
    # 3. Balanced sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_xgb)
    
    # 4. XGBoost parameters
    model_xgb_P = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=9,
        learning_rate=0.03,
        objective="multi:softprob",
        num_class=n_pt,
        random_state=SEED,
        tree_method="exact",
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=1.0,
        reg_alpha=0.5,
        reg_lambda=3.0,
        n_jobs=-1
    )
    
    model_xgb_P.fit(
        X_train_xgb, y_train_xgb,
        sample_weight=sample_weights,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=False
    )
    print("XGBoost 訓練完成！直接生成驗證集預測結果...")
    xgb_val_preds = model_xgb_P.predict(X_val_xgb)

    # ---------------------------------------------------------
    # 如果 pretraining 做了，繼續訓練 Action & Rally for remaining epochs
    # ---------------------------------------------------------
    print("--- 開始 PyTorch 獨立模型同步訓練 (Action & Rally) ---")
    for ep in range(1, args.epochs+1):
        model_A.train(); model_R.train()
        run_loss = 0.0
        for Xb, yAb, yPb, yRb, Lb in train_loader:
            Xb, yAb, yPb, yRb, Lb = Xb.to(device), yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
            
            opt_A.zero_grad(); opt_R.zero_grad()
            
            la = model_A(Xb, Lb)
            lr = model_R(Xb, Lb)
            
            loss_A = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
            loss_R = bce_rally(lr, yRb)
            
            # 調整損失權重分配給 Action 與 Rally
            loss = 0.75 * loss_A + 0.25 * loss_R
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_R.parameters(), 1.0)
            
            opt_A.step(); opt_R.step()
            run_loss += loss.item() * Xb.size(0)

        model_A.eval(); model_R.eval()
        val_loss = 0.0
        allA, allAp, allR, allRp = [], [], [], []
        with torch.no_grad():
            for Xb, yAb, yPb, yRb, Lb in val_loader:
                Xb, yAb, yPb, yRb, Lb = Xb.to(device), yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
                
                la = model_A(Xb, Lb)
                lr = model_R(Xb, Lb)
                
                loss_A = ce_action(la.view(-1, la.size(-1)), yAb.view(-1))
                loss_R = bce_rally(lr, yRb)
                
                loss = 0.75 * loss_A + 0.25 * loss_R
                val_loss += loss.item() * Xb.size(0)

                allR += yRb.detach().cpu().tolist(); allRp += torch.sigmoid(lr).detach().cpu().tolist()
                yA_flat = yAb.view(-1).detach().cpu().numpy()
                a_pred = la.argmax(-1).view(-1).detach().cpu().numpy()
                
                mA = (yA_flat != -1)
                allA += yA_flat[mA].tolist(); allAp += a_pred[mA].tolist()

        tr_loss = run_loss / len(train_loader.dataset); va_loss = val_loss / len(val_loader.dataset)
        
        # 評估分數計算 (PointModel F1 直接帶入 XGBoost 的靜態結果)
        try:
            f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
            f1P = f1_score(y_val_xgb, xgb_val_preds, average="macro") if len(y_val_xgb) else 0.0
            auc = roc_auc_score(allR, allRp) if len(set(allR)) > 1 else 0.5
        except Exception:
            f1A, f1P, auc = 0.0, 0.0, 0.5
            
        final = 0.4 * f1A + 0.4 * f1P + 0.2 * auc
        print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point(XGB)={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")

    print("--- 開始對測試集進行混合模型推論 ---")
    
    # 預先處理測試集的時序特徵 (保持與訓練集相同的特徵工程與編碼方式)
    test_xgb_base = test.copy()
    for col in FEATURES:
        dtype = pd.CategoricalDtype(categories=cats[col])
        test_xgb_base[f"{col}_enc"] = test_xgb_base[col].astype(dtype).cat.codes + 1
        
    for col in FEATURES:
        test_xgb_base[f"{col}_enc_lag1"] = test_xgb_base.groupby("rally_uid")[f"{col}_enc"].shift(1)
        test_xgb_base[f"{col}_enc_lag2"] = test_xgb_base.groupby("rally_uid")[f"{col}_enc"].shift(2)

    # Extract LSTM hidden vectors for test set (per-timestep), then align and pick last stroke per rally
    print("Extracting LSTM hidden vectors for test set (this may take time)...")
    action_matrix_test, rally_matrix_test = extract_lstm_hidden_vectors(
        test, model_A, model_R, cats, FEATURES, MAXLEN, device,
        action_feat_dim=args.action_feat_dim, rally_feat_dim=args.rally_feat_dim
    )
    test_loc = test.copy().reset_index(drop=True)
    test_loc["key"] = test_loc["rally_uid"].astype(str) + "_" + test_loc["strikeNumber"].astype(str)
    if action_matrix_test.shape[0] != len(test_loc):
        print("Warning: mismatch in test hidden extraction. Filling zeros.")
        for i in range(args.action_feat_dim):
            test_xgb_base[f"action_h_{i}"] = 0.0
        for j in range(args.rally_feat_dim):
            test_xgb_base[f"rally_h_{j}"] = 0.0
    else:
        for i in range(args.action_feat_dim):
            test_loc[f"action_h_{i}"] = action_matrix_test[:, i]
        for j in range(args.rally_feat_dim):
            test_loc[f"rally_h_{j}"] = rally_matrix_test[:, j]
        test_xgb_base = test_xgb_base.merge(test_loc[["key"] + [f"action_h_{i}" for i in range(args.action_feat_dim)] + [f"rally_h_{j}" for j in range(args.rally_feat_dim)]],
                                            left_on=test_xgb_base["rally_uid"].astype(str) + "_" + test_xgb_base["strikeNumber"].astype(str),
                                            right_on="key", how="left")
        # fillna and cleanup
        for i in range(args.action_feat_dim):
            test_xgb_base[f"action_h_{i}"] = test_xgb_base[f"action_h_{i}"].fillna(0.0)
        for j in range(args.rally_feat_dim):
            test_xgb_base[f"rally_h_{j}"] = test_xgb_base[f"rally_h_{j}"].fillna(0.0)
        test_xgb_base = test_xgb_base.drop(columns=["key"])

    pred_rows = []
    model_A.eval(); model_R.eval()
    
    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            # 1. PyTorch 模型推論 (Action & Rally)
            Xg = encode_frame_with_unk(g)
            Xp, T = pad2d_cap(Xg, MAXLEN)
            X_t = torch.tensor(Xp[None, ...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)
            
            la = model_A(X_t, L_t)
            lr = model_R(X_t, L_t)
            
            last_t = L_t.item() - 1
            a_idx = int(torch.argmax(la[0, last_t]).item())
            s_prob = float(torch.sigmoid(lr).item())
            action_pred = int(act_classes[a_idx]) if a_idx < len(act_classes) else int(act_classes[-1])
            
            # 2. XGBoost 模型推論 (Point) with LSTM hidden vectors
            g_xgb = test_xgb_base[test_xgb_base["rally_uid"] == rid].sort_values("strikeNumber")
            last_stroke_xgb = g_xgb.iloc[[-1]][xgb_features].copy()
            
            # Ensure columns exist and are numeric
            last_stroke_xgb = last_stroke_xgb.fillna(0.0)
            p_xgb_pred_idx = int(model_xgb_P.predict(last_stroke_xgb)[0])
            point_pred = int(pt_classes[p_xgb_pred_idx])
            
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
    print(f"成功儲存強效混合模型預測提交檔案至: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="inputs/train.csv")
    ap.add_argument("--test", default="inputs/test_new.csv")
    ap.add_argument("--sample", default="result/sample_submission.csv")
    ap.add_argument("--out", default="result/submission.csv")
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256, help="Action LSTM hidden size")
    ap.add_argument("--hidden_rally", type=int, default=128, help="Rally LSTM hidden size")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--cap", type=int, default=11)
    ap.add_argument("--bidirectional", type=bool, default=False)
    ap.add_argument("--pretrain_epochs", type=int, default=3, help="短暫預訓練 LSTM 的 epoch 數，以便抽取有意義的 hidden features")
    ap.add_argument("--action_feat_dim", type=int, default=256, help="Action hidden vector dims to include in XGBoost (default 256)")
    ap.add_argument("--rally_feat_dim", type=int, default=128, help="Rally hidden vector dims to include in XGBoost (default 128)")
    args = ap.parse_args()
    
    MAXLEN = args.max_len
    main(args)
