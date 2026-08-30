# run_decision_tree_baseline.py
"""
Decision Tree baseline for Multi-Task Shot Prediction.
Converts sequential data into tabular frames with Lag features, score normalization,
and Rally length flags, then fits Decision Tree Classifiers.
"""

import argparse
import random
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier  # 引入決策樹模型

# -------------------------
# 再現性設定
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# 分數正規化與特徵工程
# -------------------------
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
    if is_inference:
        rally_counts = rally_counts + 1
        
    df["rally_total_count"] = df["rally_uid"].map(rally_counts)
    df["rally_in_5"] = (df["rally_total_count"] <= 5).astype(int)
    df["rally_6_15"] = ((df["rally_total_count"] >= 6) & (df["rally_total_count"] <= 15)).astype(int)
    df["rally_long"] = (df["rally_total_count"] >= 16).astype(int)
    df = df.drop(columns=["rally_total_count"])
    return df

def add_lag_features(df):
    grouped = df.groupby("rally_uid")
    attr_map = {
        "actionId": "action",
        "pointId": "point",
        "positionId": "pos",
        "strikeId": "strike",
        "handId": "hand",
        "strengthId": "strength",
        "spinId": "spin"
    }
    prefixes = ["prev_", "prev2_", "prev3_", "prev4_"]
    
    for shift_amt, prefix in enumerate(prefixes, start=1):
        for orig_col, short_name in attr_map.items():
            col_name = f"{prefix}{short_name}"
            df[col_name] = grouped[orig_col].shift(shift_amt).fillna(0).astype(int)
            
    # 時間序列遮罩防禦 (Masking)
    mask_strike_1 = (df["strikeNumber"] == 1)
    all_lag_cols = [f"{p}{short}" for p in prefixes for short in attr_map.values()]
    df.loc[mask_strike_1, all_lag_cols] = 0
    
    mask_strike_2 = (df["strikeNumber"] == 2)
    lag234_cols = [f"{p}{short}" for p in ["prev2_", "prev3_", "prev4_"] for short in attr_map.values()]
    df.loc[mask_strike_2, lag234_cols] = 0
    
    mask_strike_3 = (df["strikeNumber"] == 3)
    lag34_cols = [f"{p}{short}" for p in ["prev3_", "prev4_"] for short in attr_map.values()]
    df.loc[mask_strike_3, lag34_cols] = 0
    
    mask_strike_4 = (df["strikeNumber"] == 4)
    lag4_cols = [f"{p}{short}" for p in ["prev4_"] for short in attr_map.values()]
    df.loc[mask_strike_4, lag4_cols] = 0
    
    return df

# -------------------------
# 資料集結構轉換函數 (Sequence -> Tabular)
# -------------------------
def build_tabular_dataset(df, X_encoded):
    """
    將序列資料攤平成適合決策樹訓練的 2D Tabular 矩陣。
    對於長度為 N 的 Rally，特徵取自第 0 到 N-2 拍，目標預測值取自第 1 到 N-1 拍。
    """
    X_rows, yA_rows, yP_rows, yR_rows = [], [], [], []
    
    for rid, g in df.groupby("rally_uid"):
        if len(g) < 2: 
            continue
        # 抓取該 group 在原始 DataFrame 中的位置索引
        idxs = g.index.values
        
        # 特徵 (t) 與預測目標 (t+1) 對齊
        X_rows.append(X_encoded[idxs[:-1]])
        yA_rows.append(g["actionId"].values[1:])
        yP_rows.append(g["pointId"].values[1:])
        yR_rows.append(g["serverGetPoint"].values[1:])
        
    return np.vstack(X_rows), np.concatenate(yA_rows), np.concatenate(yP_rows), np.concatenate(yR_rows)

# -------------------------
# 主程式
# -------------------------
def main(args):
    class Tee(object):
        def __init__(self, *files): self.files = files
        def write(self, obj):
            for f in self.files: f.write(obj); f.flush()
        def flush(self):
            for f in self.files: f.flush()

    log_file = open("result/log_dt.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    # 1. 讀取資料
    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample) if args.sample else pd.DataFrame()

    # 2. 特徵工程與多拍資訊擴充
    train = normalize_scores(train, cap=args.cap)
    test  = normalize_scores(test, cap=args.cap)
    train = add_rally_length_features(train, is_inference=False)
    test  = add_rally_length_features(test, is_inference=True)
    train = add_lag_features(train)
    test  = add_lag_features(test)

    # 定義特徵清單 (包含先前擴充的 28 個歷史標籤)
    base_features = [
        "sex","handId","strengthId","spinId",
        "pointId","actionId","positionId","strikeId",
        "scoreSelf_capped","scoreOther_capped","score_diff","is_deuce","strikeNumber",
        "rally_in_5", "rally_6_15", "rally_long"
    ]
    prefixes = ["prev_", "prev2_", "prev3_", "prev4_"]
    short_names = ["action", "point", "pos", "strike", "hand", "strength", "spin"]
    lag_features = [f"{p}{s}" for p in prefixes for s in short_names]
    
    FEATURES = base_features + lag_features

    # 3. 類別編碼器 (與舊 Baseline 相同，確保未知值自動轉為 UNK)
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

    # 進行全域特徵編碼
    train_encoded = encode_frame_with_unk(train)
    test_encoded  = encode_frame_with_unk(test)

    # 4. 驗證集切分 (以 rally_uid 為主體進行切分，防止同一個回合內的拍步交叉洩漏)
    unique_rallies = train["rally_uid"].unique()
    rally_outcomes = train.groupby("rally_uid")["serverGetPoint"].first().values
    
    tr_rallies, va_rallies = train_test_split(
        unique_rallies, test_size=args.val_size, random_state=SEED, stratify=rally_outcomes
    )
    tr_rallies_set = set(tr_rallies)
    
    df_tr = train[train["rally_uid"].isin(tr_rallies_set)].copy()
    df_va = train[~train["rally_uid"].isin(tr_rallies_set)].copy()

    # 5. 建構表格化訓練與驗證矩陣
    print("正在建構決策樹平坦化特徵矩陣...")
    X_tr, yA_tr, yP_tr, yR_tr = build_tabular_dataset(df_tr, train_encoded)
    X_va, yA_va, yP_va, yR_va = build_tabular_dataset(df_va, train_encoded)

    # 目標 ID 對齊與轉換映射
    act_classes = np.sort(train["actionId"].unique())
    pt_classes  = np.sort(train["pointId"].unique())
    act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}
    
    yA_tr = np.vectorize(lambda v: act_id2idx.get(v, -1))(yA_tr)
    yP_tr = np.vectorize(lambda v: pt_id2idx.get(v, -1))(yP_tr)
    yA_va = np.vectorize(lambda v: act_id2idx.get(v, -1))(yA_va)
    yP_va = np.vectorize(lambda v: pt_id2idx.get(v, -1))(yP_va)

    # 6. 初始化與訓練決策樹模型 (為了解決類別不平衡問題，加入 class_weight="balanced")
    print(f"正在訓練決策樹模型 (最大深度={args.max_depth}, 葉節點最小樣本數={args.min_samples_leaf})...")
    
    model_A = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, 
                                     class_weight="balanced", random_state=SEED)
    model_P = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, 
                                     class_weight="balanced", random_state=SEED)
    model_R = DecisionTreeClassifier(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf, 
                                     class_weight="balanced", random_state=SEED)

    model_A.fit(X_tr, yA_tr)
    model_P.fit(X_tr, yP_tr)
    model_R.fit(X_tr, yR_tr)
    print("模型訓練完成！")

    # 7. 驗證集評估
    pred_A_va = model_A.predict(X_va)
    pred_P_va = model_P.predict(X_va)
    pred_R_va = model_R.predict_proba(X_va)[:, 1] # 取得類別 1 的機率值

    try:
        f1A = f1_score(yA_va, pred_A_va, average="macro")
        f1P = f1_score(yP_va, pred_P_va, average="macro")
        auc = roc_auc_score(yR_va, pred_R_va) if len(set(yR_va)) > 1 else 0.5
    except Exception:
        f1A, f1P, auc = 0.0, 0.0, 0.5
        
    final = 0.4 * f1A + 0.4 * f1P + 0.2 * auc
    print(f"\n[決策樹驗證表現] F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")

    # 8. 測試集推論 (抓取每個回合的最後已知拍步資訊)
    print("\n正在對測試集進行最終拍步推論...")
    last_shot_info = test.reset_index().groupby("rally_uid", as_index=False)["index"].last()
    last_shot_idxs = last_shot_info["index"].values
    test_rallies   = last_shot_info["rally_uid"].values

    # 擷取最後一拍的特徵矩陣
    X_test = test_encoded[last_shot_idxs]

    # 模型預測
    pred_A_idx = model_A.predict(X_test)
    pred_P_idx = model_P.predict(X_test)
    pred_R_prob = model_R.predict_proba(X_test)[:, 1]

    # 將預測索引映射回原始的類別標籤 ID
    action_preds = act_classes[pred_A_idx]
    point_preds  = pt_classes[pred_P_idx]

    # 組合提交格式
    pred_df = pd.DataFrame({
        "rally_uid": test_rallies,
        "actionId": action_preds,
        "pointId": point_preds,
        "serverGetPoint": pred_R_prob
    })

    if len(sample) > 0 and 'rally_uid' in sample.columns:
        out = sample.drop(columns=["serverGetPoint","pointId","actionId"], errors="ignore").merge(pred_df, on="rally_uid", how="left")
    else:
        out = pred_df

    cols_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out.reindex(columns=[c for c in cols_order if c in out.columns])
    out.to_csv(args.out, index=False)
    print(f"推論完成！決策樹模型預測結果已儲存至: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="inputs/train.csv")
    ap.add_argument("--test", default="inputs/test_new.csv")
    ap.add_argument("--sample", default="result/sample_submission.csv")
    ap.add_argument("--out", default="result/submission.csv")
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--cap", type=int, default=11)
    
    # 💡 決策樹專用超參數定義 (保留原參數防自動腳本出錯)
    ap.add_argument("--max_depth", type=int, default=12, help="決策樹最大深度")
    ap.add_argument("--min_samples_leaf", type=int, default=4, help="葉節點最小樣本數")
    
    # 保留舊深度學習參數空殼以防命令列報錯
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--bidirectional", type=bool, default=True)
    
    args = ap.parse_args()
    main(args)