import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

# 這些是類別特徵 (包含我們新加的 my_bh_pos, opp_bh_pos, my_style, opp_style)
CAT_FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "actionId", "positionId", "pointId", "strikeId",
    "gamePlayerId", "gamePlayerOtherId",
    "prev_action", "prev_point", "prev_pos",
    "prev2_action", "prev2_point", "prev2_pos",
    "my_bh_pos", "opp_bh_pos", 
    "my_style", "opp_style"
]

# 這些是連續數值特徵 (包含我們新加的 my_mid_ratio, opp_mid_ratio)
NUM_FEATURES = [
    "scoreSelf", "scoreOther", "strikeNumber", "score_diff",
    "my_mid_ratio", "opp_mid_ratio"
]

FEATURES = CAT_FEATURES + NUM_FEATURES

def create_player_profiles(train_df, test_df):
    """
    根據精階桌球戰術邏輯，建立選手慣用手與球風特徵
    """
    print("👤 正在建立進階選手身分證 (慣用手與球風分析)...")
    
    # 🌟 前提要求 3：排除 positionId 為 0 (無法判斷) 的無效資料
    valid_pos_df = train_df[train_df["positionId"].isin([1, 2, 3])].copy()

    # ==========================================
    # 🌟 邏輯 1：判斷左右手 (只看使用「反手」時的常用站位)
    # 假設 handId == 2 代表反手。排除站位 2 (中間)。
    # ==========================================
    bh_df = valid_pos_df[(valid_pos_df["handId"] == 2) & (valid_pos_df["positionId"].isin([1, 3]))]
    
    # 計算每個選手打反手時，最常站在 1(左) 還是 3(右)
    player_bh_pos = bh_df.groupby("gamePlayerId")["positionId"].agg(
        lambda x: x.mode()[0] if not x.mode().empty else np.nan
    ).to_dict()
    
    # 全局反手預設站位 (防冷啟動：通常右撇子多，反手站位多為 1 左邊)
    global_bh_pos = bh_df["positionId"].mode()[0] if not bh_df.empty else 1

    # ==========================================
    # 🌟 邏輯 2：判斷正/反手球風 (站位偏中 vs. 偏兩側)
    # 統計每個選手在有效站位中，站在 2(中間) 的比例
    # ==========================================
    # 計算每位選手在 1, 2, 3 站位的次數
    pos_counts = valid_pos_df.groupby("gamePlayerId")["positionId"].value_counts().unstack(fill_value=0)
    
    # 確保 1, 2, 3 欄位都存在
    for col in [1, 2, 3]:
        if col not in pos_counts.columns:
            pos_counts[col] = 0
            
    pos_counts["total"] = pos_counts[1] + pos_counts[2] + pos_counts[3]
    # 計算「中間站位比例 (Middle Ratio)」
    pos_counts["middle_ratio"] = pos_counts[2] / (pos_counts["total"] + 1e-5) # 加微小值防分母為0
    
    player_mid_ratio = pos_counts["middle_ratio"].to_dict()
    global_mid_ratio = pos_counts["middle_ratio"].median() # 全局中位數基準

    # 設定 Threshold (以全局中位數為界線，或你也可以自訂如 0.3)
    # 比例大於基準 -> 1 (反手依賴型/站中間)
    # 比例小於基準 -> 0 (正手跑動型/站兩側)
    player_style = {k: (1 if v > global_mid_ratio else 0) for k, v in player_mid_ratio.items()}

    # ==========================================
    # 映射特徵回 Train 與 Test 資料集
    # ==========================================
    for df in [train_df, test_df]:
        # 慣用手特徵 (類別型：1 或 3)
        df["my_bh_pos"] = df["gamePlayerId"].map(player_bh_pos).fillna(global_bh_pos).astype(int)
        df["opp_bh_pos"] = df["gamePlayerOtherId"].map(player_bh_pos).fillna(global_bh_pos).astype(int)
        
        # 球風特徵 - 連續數值 (XGBoost 最喜歡這個，能精細切分)
        df["my_mid_ratio"] = df["gamePlayerId"].map(player_mid_ratio).fillna(global_mid_ratio).astype(float)
        df["opp_mid_ratio"] = df["gamePlayerOtherId"].map(player_mid_ratio).fillna(global_mid_ratio).astype(float)
        
        # 球風特徵 - 類別型 (超過 Threshold 標記為 1，否則 0)
        df["my_style"] = df["gamePlayerId"].map(player_style).fillna(0).astype(int)
        df["opp_style"] = df["gamePlayerOtherId"].map(player_style).fillna(0).astype(int)

    return train_df, test_df

def create_features(df):
    df["score_diff"] = df["scoreSelf"] - df["scoreOther"]
    df["prev_action"] = df.groupby("rally_uid")["actionId"].shift(1).fillna(0).astype(int)
    df["prev_point"]  = df.groupby("rally_uid")["pointId"].shift(1).fillna(0).astype(int)
    df["prev_pos"]    = df.groupby("rally_uid")["positionId"].shift(1).fillna(0).astype(int)
    df["prev2_action"] = df.groupby("rally_uid")["actionId"].shift(2).fillna(0).astype(int)
    df["prev2_point"]  = df.groupby("rally_uid")["pointId"].shift(2).fillna(0).astype(int)
    df["prev2_pos"]    = df.groupby("rally_uid")["positionId"].shift(2).fillna(0).astype(int)
    return df

def main():
    print("📥 正在讀取資料...")
    train = pd.read_csv("dataset/train.csv").sort_values(["rally_uid", "strikeNumber"])
    test  = pd.read_csv("dataset/test_new.csv").sort_values(["rally_uid", "strikeNumber"])

    # 🌟 優化細節 1：確保 strikeNumber 的數值邊界與主模型對齊 (維持原廠設定)
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)

    # 👇 1. 建立並合併進階選手習慣特徵 (內部會自動排除 positionId 為 0 的干擾)
    train, test = create_player_profiles(train, test)

    # 2. 執行基礎時序特徵工程
    train = create_features(train)
    test  = create_features(test)

    # 3. 冷啟動防禦機制 (處理沒看過的選手)
    FREQ_THRESHOLD = 15
    for col in ["gamePlayerId", "gamePlayerOtherId"]:
        player_counts = train[col].value_counts()
        popular_players = player_counts[player_counts > FREQ_THRESHOLD].index
        train.loc[~train[col].isin(popular_players), col] = 0
        test.loc[~test[col].isin(popular_players), col] = 0
        
        # 🌟 修正致命盲點：強制將 ID 欄位統一轉為 int，確保後面 category 映射時對齊
        train[col] = train[col].astype(int)
        test[col] = test[col].astype(int)

    print("🔄 正在進行時序目標對齊 (預測下一拍)...")
    # 將預測目標設為「下一拍的落點」
    train["target_pointId"] = train.groupby("rally_uid")["pointId"].shift(-1)

    # 先刪除掉沒有「未來」可以預測的絕對最後一拍
    train_clean = train.dropna(subset=["target_pointId"]).copy()

    # 在已經乾淨的資料中，找出每一局「新的最後一拍」，並打上末拍位標記
    train_clean["is_last_shot"] = 0
    last_valid_indices = train_clean.groupby("rally_uid").tail(1).index
    train_clean.loc[last_valid_indices, "is_last_shot"] = 1

    # 標籤 0-based 連續編碼轉換
    pt_classes = np.sort(train_clean["target_pointId"].unique())
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}
    idx2pt_id = {i: v for i, v in enumerate(pt_classes)}
    train_clean["point_target"] = train_clean["target_pointId"].map(pt_id2idx)

    # 準備特徵矩陣 X 與標籤 y
    X = train_clean[FEATURES].copy()
    y = train_clean["point_target"].copy()
    
    # 🌟 關鍵對齊優化：利用 pandas 的 CategoricalDtype，強制讓 train 和 test 的類別完全共享相同的定義域
    for col in CAT_FEATURES:
        # 先抓出訓練集該欄位的所有可能類別
        categories_structure = pd.Categorical(X[col]).categories
        # 強制轉型為 category
        X[col] = pd.Categorical(X[col], categories=categories_structure)
        
    is_last_shot_array = train_clean["is_last_shot"].values 
    groups = train_clean["match"] if "match" in train_clean.columns else train_clean["gamePlayerId"]

    print("🚀 開始訓練原生類別支援版 XGBoost 模型...")
    gkf = GroupKFold(n_splits=5)
    tr_idx, va_idx = next(gkf.split(X, y, groups=groups))

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    is_last_va = is_last_shot_array[va_idx]

    sample_weights = compute_sample_weight('balanced', y_tr)

    # 調優後的穩健派樹模型參數
    xgb_model = XGBClassifier(
        n_estimators=10000, 
        max_depth=6,            # 基於你提供的實驗數據，15 層在開啟類別支援後表現極佳
        learning_rate=0.018,      # 稍微降低學習率，配合 15 層深度，防止高難度條件組合過擬合
        subsample=0.8, 
        colsample_bytree=0.8, 
        objective='multi:softmax',
        num_class=len(pt_classes), 
        random_state=42,
        tree_method='hist',
        enable_categorical=True, # 開啟 XGBoost 原生類別支援
        early_stopping_rounds=50
    )

    xgb_model.fit(
        X_tr, y_tr, 
        sample_weight=sample_weights, 
        eval_set=[(X_va, y_va)], 
        verbose=50
    )

    # =========================================================
    # 🌟 全新公平評估：模擬線上考卷的「隨機截斷抽樣」
    # =========================================================
    y_pred_all = xgb_model.predict(X_va)
    
    # 1. 抓出 Validation Set 每一筆資料對應的 rally_uid
    # (因為 X_va 是 DataFrame，我們透過 index 回 train_clean 找 rally_uid)
    va_rally_uids = train_clean.loc[X_va.index, "rally_uid"].values
    
    # 2. 建立一個暫存的 DataFrame 方便做分組抽樣
    val_eval_df = pd.DataFrame({
        "rally_uid": va_rally_uids,
        "y_true": y_va.values,
        "y_pred": y_pred_all
    })
    
    # 3. 核心邏輯：對每一個局 (rally_uid) 隨機抽取 1 拍作為考題
    # 加上 random_state 確保每次跑結果一致，方便對比
    fair_val_df = val_eval_df.groupby("rally_uid").sample(n=1, random_state=42)
    
    # 4. 計算隨機截斷的 F1_point 分數
    f1_macro_fair = f1_score(fair_val_df["y_true"], fair_val_df["y_pred"], average='macro')
    
    print("-" * 60)
    print(f"🎯 XGBoost 公平隨機截斷 F1_point 分數: {f1_macro_fair:.4f}")
    print("-" * 60)

    # =========================================================
    # 🔮 預測測試集階段
    # =========================================================
    print("🔮 正在產生測試集預測...")
    test_last_shots = test.groupby("rally_uid").tail(1).copy()
    X_test = test_last_shots[FEATURES].copy()
    
    # 🌟 核心修復：使用與訓練集「完全一模一樣」的類別架構去格式化測試集，徹底根除線上報錯
    for col in CAT_FEATURES:
        categories_structure = pd.Categorical(X[col]).categories
        X_test[col] = pd.Categorical(X_test[col], categories=categories_structure)

    # 預測並轉回官方原始 pointId 格式
    test_preds_idx = xgb_model.predict(X_test)
    test_last_shots["pointId"] = [int(idx2pt_id[idx]) for idx in test_preds_idx]

    out_df = test_last_shots[["rally_uid", "pointId"]]
    out_df.to_csv("output/train_point_only.csv", index=False)
    print("✅ 檔案已成功更新至 output/train_point_only.csv")

if __name__ == "__main__":
    main()