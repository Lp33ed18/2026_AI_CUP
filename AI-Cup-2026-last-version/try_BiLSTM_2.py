import os
import argparse
import copy
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def out_path_with_task_suffix(path: str, w_act: float, w_pt: float, w_rly: float) -> str:
    tags = []
    if w_act > 0: tags.append("act")
    if w_pt > 0: tags.append("pt")
    if w_rly > 0: tags.append("rly")
    if not tags: return path
    suffix = "_" + "_".join(tags)
    if "." in path: return f"{path.rsplit('.', 1)[0]}{suffix}.{path.rsplit('.', 1)[1]}"
    return f"{path}{suffix}"

CAT_FEATURES = [
    "sex", "handId", "strengthId", "spinId",
    "actionId", "positionId", "pointId", "strikeId",
    "gamePlayerId", "gamePlayerOtherId",
    "prev_action", "prev_point", "prev_pos",
    "prev2_action", "prev2_point", "prev2_pos",
    "my_bh_pos", "opp_bh_pos", 
    "my_style", "opp_style",
    "my_is_pimple", "opp_is_pimple",         
    "prev_point_x", "prev_point_y",          
    "prev2_point_x", "prev2_point_y"         
]

NUM_FEATURES = [
    "scoreSelf", "scoreOther", "strikeNumber", "score_diff",
    "my_mid_ratio", "opp_mid_ratio",
    "hist_len"                               
]
PAD_TOKEN = 0

class RallyDataset(Dataset):
    def __init__(self, X_cat, X_num, yA, yP, yR, L, uids=None, is_train=False, extract_mode=False):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.yA = torch.tensor(yA, dtype=torch.long)
        self.yP = torch.tensor(yP, dtype=torch.long)
        self.yR = torch.tensor(yR, dtype=torch.float32)
        self.L  = torch.tensor(L,  dtype=torch.long)
        self.uids = uids # 用於對齊 embedding
        self.is_train = is_train
        self.extract_mode = extract_mode # 🌟 新增：萃取模式不截斷

    def __len__(self): return self.X_cat.shape[0]
    
    def __getitem__(self, i):
        length = self.L[i].item()
        
        # 🌟 如果是為了萃取特徵，永遠取最後一拍 (不隨機截斷)
        if self.extract_mode:
            cut_idx = length
        else:
            if length > 1:
                if self.is_train: cut_idx = random.randint(1, length)
                else: np_rand = np.random.RandomState(i); cut_idx = np_rand.randint(1, length + 1)
            else: cut_idx = 1
            
        xc_out = self.X_cat[i].clone()
        xn_out = self.X_num[i].clone() 
        ya_out = self.yA[i].clone()
        yp_out = self.yP[i].clone()

        # 🌟 ID Dropout (冷啟動防禦)，萃取模式下不使用
        if self.is_train and not self.extract_mode and random.random() < 0.15:
            mask_indices = [8, 9, 16, 17, 18, 19, 20, 21]
            for idx in mask_indices: xc_out[:, idx] = PAD_TOKEN 
        
        if cut_idx < length:
            xc_out[cut_idx:] = PAD_TOKEN; xn_out[cut_idx:] = 0.0
            ya_out[cut_idx:] = -1; yp_out[cut_idx:] = -1
            
        return xc_out, xn_out, ya_out, yp_out, self.yR[i], torch.tensor(cut_idx, dtype=torch.long)

class MultiTaskBiLSTM(nn.Module):
    def __init__(self, num_tokens_per_feature, num_num_features, n_act, n_pt, emb_dim=32, hidden=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n+1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        raw_dim = len(num_tokens_per_feature) * emb_dim + num_num_features
        
        self.input_proj = nn.Sequential(nn.Linear(raw_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True, 
                            bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        
        self.drop = nn.Dropout(dropout)
        
        lstm_out_dim = hidden * 2
        self.act_head = nn.Sequential(nn.Linear(lstm_out_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_act))
        self.pt_head  = nn.Sequential(nn.Linear(lstm_out_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_pt))
        self.attn = nn.Linear(lstm_out_dim, 1)
        self.rly_head = nn.Linear(lstm_out_dim, 1)
        
    def forward(self, X_cat, X_num, lengths, return_embeddings=False):
        es = [emb(X_cat[:,:,i]) for i,emb in enumerate(self.embs)]
        x = torch.cat(es + [X_num], dim=-1)
        x = self.input_proj(x)
        
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(packed)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=X_cat.size(1))
        o_dropped = self.drop(o)
        
        mask = (X_cat[:,:,0]!=PAD_TOKEN)
        attn_weights = self.attn(o_dropped).squeeze(-1)
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), o_dropped).squeeze(1)
        
        # 🌟 收到萃取指令時，直接回傳隱藏層與注意力重點
        if return_embeddings:
            return o, context
        
        return self.act_head(o_dropped), self.pt_head(o_dropped), self.rly_head(context).squeeze(-1)

def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); out[:len(a)] = a; return out
def pad2d_num(a, m, pad_val=0.0):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.float32); out[:len(a)] = a; return out
def pad1d(a, m, ignore_index=-1):
    out = np.full((m,), ignore_index, dtype=np.int64); out[:len(a)] = a; return out

def main(args):
    train = pd.read_csv(args.train).sort_values(["rally_uid","strikeNumber"])
    test  = pd.read_csv(args.test).sort_values(["rally_uid","strikeNumber"])
    train["strikeNumber"] = train["strikeNumber"].clip(0, 40)
    test["strikeNumber"]  = test["strikeNumber"].clip(0, 40)

    def create_player_profiles(train_df, test_df):
        # 🌟 啟動上帝視角：將 Train 和 Test 疊起來一起看
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # 1. 抓出所有左/右撇子 (從 all_df 找，涵蓋 Test 的 31 位新選手)
        valid_pos_df = all_df[all_df["strikeNumber"].isin([1, 2]) & all_df["positionId"].isin([1, 3])]
        player_fav_pos = valid_pos_df.groupby("gamePlayerId")["positionId"].agg(lambda x: x.mode()[0] if not x.mode().empty else 1).to_dict()
        global_fav_pos = 1 

        # 2. 統計所有選手的站位依賴度 (從 all_df 找)
        pos_df_all = all_df[all_df["strikeNumber"].isin([1, 2]) & all_df["positionId"].isin([1, 2, 3])]
        pos_counts = pos_df_all.groupby("gamePlayerId")["positionId"].value_counts().unstack(fill_value=0)
        for col in [1, 2, 3]:
            if col not in pos_counts.columns: pos_counts[col] = 0
        pos_counts["total"] = pos_counts[1] + pos_counts[2] + pos_counts[3]
        pos_counts["middle_ratio"] = pos_counts[2] / (pos_counts["total"] + 1e-5)
        
        player_mid_ratio = pos_counts["middle_ratio"].to_dict()
        global_mid_ratio = pos_counts["middle_ratio"].median()
        player_style = {k: (1 if v > global_mid_ratio else 0) for k, v in player_mid_ratio.items()}
        
        # 3. 抓出所有顆粒拍選手 (從 all_df 找，即使他在 Test 才第一次拿出來)
        pimple_players = set(all_df[all_df["actionId"].isin([8, 9])]["gamePlayerId"].unique())

        # 4. 把全知全能的統計結果，分別貼回 train_df 和 test_df
        for df in [train_df, test_df]:
            df["my_bh_pos"] = df["gamePlayerId"].map(player_fav_pos).fillna(global_fav_pos).astype(int)
            df["opp_bh_pos"] = df["gamePlayerOtherId"].map(player_fav_pos).fillna(global_fav_pos).astype(int)
            df["my_mid_ratio"] = df["gamePlayerId"].map(player_mid_ratio).fillna(global_mid_ratio).astype(float)
            df["opp_mid_ratio"] = df["gamePlayerOtherId"].map(player_mid_ratio).fillna(global_mid_ratio).astype(float)
            df["my_style"] = df["gamePlayerId"].map(player_style).fillna(0).astype(int)
            df["opp_style"] = df["gamePlayerOtherId"].map(player_style).fillna(0).astype(int)
            df["my_is_pimple"] = df["gamePlayerId"].apply(lambda x: 1 if x in pimple_players else 0).astype(int)
            df["opp_is_pimple"] = df["gamePlayerOtherId"].apply(lambda x: 1 if x in pimple_players else 0).astype(int)
            
        return train_df, test_df

    train, test = create_player_profiles(train, test)

    def create_features(df):
        df["score_diff"] = df["scoreSelf"] - df["scoreOther"]
        df["prev_action"] = df.groupby("rally_uid")["actionId"].shift(1).fillna(0).astype(int)
        df["prev_point"]  = df.groupby("rally_uid")["pointId"].shift(1).fillna(0).astype(int)
        df["prev_pos"]    = df.groupby("rally_uid")["positionId"].shift(1).fillna(0).astype(int)
        df["prev2_action"] = df.groupby("rally_uid")["actionId"].shift(2).fillna(0).astype(int)
        df["prev2_point"]  = df.groupby("rally_uid")["pointId"].shift(2).fillna(0).astype(int)
        df["prev2_pos"]    = df.groupby("rally_uid")["positionId"].shift(2).fillna(0).astype(int)
        def get_x(p): return (p - 1) % 3 if p > 0 else -1
        def get_y(p): return (p - 1) // 3 if p > 0 else -1
        df["prev_point_x"] = df["prev_point"].apply(get_x).astype(int)
        df["prev_point_y"] = df["prev_point"].apply(get_y).astype(int)
        df["prev2_point_x"] = df["prev2_point"].apply(get_x).astype(int)
        df["prev2_point_y"] = df["prev2_point"].apply(get_y).astype(int)
        df["hist_len"] = (df["strikeNumber"] - 1).clip(0, 10).astype(int)
        return df

    train = create_features(train)
    test  = create_features(test)

    cats_dict = {c: pd.Categorical(train[c]).categories for c in CAT_FEATURES}
    def encode_cat_frame(df):
        outs = []
        for col in CAT_FEATURES:
            codes = pd.Categorical(df[col], categories=cats_dict[col]).codes + 1
            outs.append(np.asarray(codes, dtype=np.int64))
        return np.stack(outs, axis=1)

    def encode_num_frame(df): return df[NUM_FEATURES].values.astype(np.float32)

    # 🌟 確保在這裡存下 uid_list
    Xc_list, Xn_list, yA_list, yP_list, yR_list, L_list, match_list, uid_list = [], [], [], [], [], [], [], []
    for rid, g in train.groupby("rally_uid"):
        if len(g) < 2: continue
        Xc = encode_cat_frame(g)[:-1]; Xn = encode_num_frame(g)[:-1] 
        yA = g["actionId"].values[1:].astype(np.int64); yP = g["pointId"].values[1:].astype(np.int64)
        Xc_list.append(Xc); Xn_list.append(Xn)
        yA_list.append(yA); yP_list.append(yP)
        yR_list.append(int(g["serverGetPoint"].iloc[0])); L_list.append(len(Xc))
        match_list.append(g["match"].iloc[0] if "match" in g.columns else rid)
        uid_list.append(rid)

    MAXLEN = max(L_list)
    Xc_all  = np.stack([pad2d(s, MAXLEN) for s in Xc_list]); Xn_all  = np.stack([pad2d_num(s, MAXLEN) for s in Xn_list]) 
    yA_all  = np.stack([pad1d(s, MAXLEN) for s in yA_list]); yP_all  = np.stack([pad1d(s, MAXLEN) for s in yP_list])
    yR_all  = np.array(yR_list, dtype=np.float32); L_all   = np.array(L_list, dtype=np.int64)

    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes); act_id2idx = {v:i for i,v in enumerate(act_classes)}
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes);  pt_id2idx  = {v:i for i,v in enumerate(pt_classes)}
    yA_all = np.vectorize(act_id2idx.get)(yA_all, -1); yP_all = np.vectorize(pt_id2idx.get)(yP_all, -1)

    gkf = GroupKFold(n_splits=5)
    tr_idx, va_idx = next(gkf.split(Xc_all, yR_all, groups=match_list))
    
    Xc_tr, Xc_va = Xc_all[tr_idx], Xc_all[va_idx]; Xn_tr, Xn_va = Xn_all[tr_idx], Xn_all[va_idx]
    yA_tr, yA_va = yA_all[tr_idx], yA_all[va_idx]; yP_tr, yP_va = yP_all[tr_idx], yP_all[va_idx]
    yR_tr, yR_va = yR_all[tr_idx], yR_all[va_idx]; L_tr,  L_va  = L_all[tr_idx],  L_all[va_idx]

    act_counts = np.bincount(yA_tr[yA_tr!=-1].ravel(), minlength=n_act) + 1
    pt_counts  = np.bincount(yP_tr[yP_tr!=-1].ravel(), minlength=n_pt) + 1
    act_w = torch.tensor(1.0/np.sqrt(act_counts), dtype=torch.float32); act_w = (act_w * (n_act/act_w.sum()))
    
    train_ds = RallyDataset(Xc_tr, Xn_tr, yA_tr, yP_tr, yR_tr, L_tr, is_train=True)
    val_ds   = RallyDataset(Xc_va, Xn_va, yA_va, yP_va, yR_va, L_va, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=max(args.batch*2,128), shuffle=False)

    num_tokens_per_feature = [len(cats_dict[c]) + 1 for c in CAT_FEATURES]
    num_num_features = len(NUM_FEATURES)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskBiLSTM(
        num_tokens_per_feature, num_num_features, n_act, n_pt, 
        emb_dim=args.emb, hidden=args.hidden, num_layers=args.layers, dropout=args.drop
    ).to(device)
    
    ce_action = nn.CrossEntropyLoss(ignore_index=-1, weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(ignore_index=-1)
    bce_rally = nn.BCEWithLogitsLoss()
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    best_final = -1e18
    best_epoch = 0
    best_state = None
    
    print("🚀 Bi-LSTM 訓練開始 (雙向時間序列 + 特徵投影)...")
    
    for ep in range(1, args.epochs+1):
        model.train(); run_loss=0.0
        for Xcb, Xnb, yAb, yPb, yRb, Lb in train_loader:
            Xcb, Xnb = Xcb.to(device), Xnb.to(device)
            yAb, yPb, yRb, Lb = yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
            
            opt.zero_grad()
            la, lp, lr = model(Xcb, Xnb, Lb)
            
            loss = 0.0
            if args.w_act > 0: loss += args.w_act * ce_action(la.view(-1,la.size(-1)), yAb.view(-1))
            if args.w_pt > 0:  loss += args.w_pt * ce_point(lp.view(-1,lp.size(-1)), yPb.view(-1))
            if args.w_rly > 0: loss += args.w_rly * bce_rally(lr, yRb)
                
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            run_loss += loss.item()*Xcb.size(0)

        model.eval(); val_loss=0.0
        allA,allAp,allP,allPp,allR,allRp=[],[],[],[],[],[]
        with torch.no_grad():
            for Xcb, Xnb, yAb, yPb, yRb, Lb in val_loader:
                Xcb, Xnb = Xcb.to(device), Xnb.to(device)
                yAb, yPb, yRb, Lb = yAb.to(device), yPb.to(device), yRb.to(device), Lb.to(device)
                
                la, lp, lr = model(Xcb, Xnb, Lb)
                
                a_pred = la.argmax(-1).detach().cpu().numpy()
                p_pred = lp.argmax(-1).detach().cpu().numpy()
                yAb_np = yAb.detach().cpu().numpy()
                yPb_np = yPb.detach().cpu().numpy()
                Lb_np  = Lb.detach().cpu().numpy()

                if args.w_rly > 0:
                    allR+=yRb.detach().cpu().tolist(); allRp+=torch.sigmoid(lr).detach().cpu().tolist()

                for i in range(Xcb.size(0)):
                    last_t = Lb_np[i] - 1
                    if last_t >= 0:
                        ya_true, yp_true = yAb_np[i, last_t], yPb_np[i, last_t]
                        ya_pred, yp_pred = a_pred[i, last_t], p_pred[i, last_t]
                        if ya_true != -1 and args.w_act > 0:
                            allA.append(ya_true); allAp.append(ya_pred)
                        if yp_true != -1 and args.w_pt > 0:
                            allP.append(yp_true); allPp.append(yp_pred)

        tr_loss = run_loss/len(train_loader.dataset)
        try:
            f1A = f1_score(allA,allAp,average="macro") if (len(allA) and args.w_act > 0) else 0.0
            f1P = f1_score(allP,allPp,average="macro") if (len(allP) and args.w_pt > 0) else 0.0
            auc = roc_auc_score(allR,allRp) if (len(set(allR))>1 and args.w_rly > 0) else 0.5
        except Exception: 
            f1A, f1P, auc = 0.0, 0.0, 0.5
            
        final = args.w_act * f1A + args.w_pt * f1P + args.w_rly * auc
        current_lr = opt.param_groups[0]['lr']
        print(f"[Epoch {ep}/{args.epochs}] LR={current_lr:.6f} F1_act={f1A:.4f} F1_pt={f1P:.4f} AUC={auc:.4f} Custom_Final~{final:.4f}")
        
        # Scheduler 原本沒有寫在迴圈裡，手動降低 lr 取代 scheduler
        if ep in [11, 19]:
            for param_group in opt.param_groups:
                param_group['lr'] *= 0.5
        
        if final > best_final:
            best_final = float(final)
            best_epoch = ep
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is not None: model.load_state_dict(best_state, strict=True)
    model.eval()
    print(f"[Best] epoch={best_epoch} best_final={best_final:.6f}")

    # ==========================================
    # 🌟 究極特徵擷取：導出 LSTM 腦波 (Embeddings)
    # ==========================================
    print("🧠 正在擷取神經網路的高維度特徵 (Embeddings) 以供 Stacking 使用...")
    
    extract_ds = RallyDataset(Xc_all, Xn_all, yA_all, yP_all, yR_all, L_all, uids=uid_list, is_train=False, extract_mode=True)
    extract_loader = DataLoader(extract_ds, batch_size=args.batch, shuffle=False)
    
    train_embeddings = []
    with torch.no_grad():
        for Xcb, Xnb, _, _, _, Lb in extract_loader:
            Xcb, Xnb = Xcb.to(device), Xnb.to(device)
            o, context = model(Xcb, Xnb, Lb.to(device), return_embeddings=True)
            
            for i in range(Xcb.size(0)):
                last_t = Lb[i].item() - 1
                combined_emb = torch.cat([o[i, last_t], context[i]], dim=-1).cpu().numpy()
                train_embeddings.append(combined_emb)
    
    train_emb_df = pd.DataFrame(train_embeddings)
    train_emb_df["rally_uid"] = uid_list
    os.makedirs("output", exist_ok=True)
    train_emb_df.to_csv("output/train_embeddings.csv", index=False)
    print("✅ 訓練集 Embeddings 儲存完畢！")

    # ==========================================
    # 推論階段 (同時導出 Test Embeddings)
    # ==========================================
    def pad2d_cap(a, m, pad_val=PAD_TOKEN):
        out = np.full((m, a.shape[1]), pad_val, dtype=np.int64); T = min(len(a), m); out[:T]=a[:T]; return out, T
    def pad2d_num_cap(a, m, pad_val=0.0):
        out = np.full((m, a.shape[1]), pad_val, dtype=np.float32); T = min(len(a), m); out[:T]=a[:T]; return out

    pred_rows = []
    test_embeddings = []
    
    with torch.no_grad():
        for rid,g in test.groupby("rally_uid"):
            Xc_g = encode_cat_frame(g); Xc_p, T = pad2d_cap(Xc_g, MAXLEN)
            Xn_g = encode_num_frame(g); Xn_p = pad2d_num_cap(Xn_g, MAXLEN)
            
            Xc_t = torch.tensor(Xc_p[None,...], dtype=torch.long, device=device)
            Xn_t = torch.tensor(Xn_p[None,...], dtype=torch.float32, device=device)
            L_t  = torch.tensor([max(1,T)], dtype=torch.long, device=device)
            
            la, lp, lr = model(Xc_t, Xn_t, L_t); last_t = L_t.item()-1
            a_idx = int(torch.argmax(la[0,last_t]).item()); p_idx = int(torch.argmax(lp[0,last_t]).item())
            s_prob = float(torch.sigmoid(lr).item())
            
            pred_rows.append({
                "rally_uid": int(rid),
                "serverGetPoint": 1 if s_prob >= 0.5 else 0,
                "pointId": int(pt_classes[p_idx]),
                "actionId": int(act_classes[a_idx])
            })
            
            # 🌟 擷取測試集的 Embeddings
            o, context = model(Xc_t, Xn_t, L_t, return_embeddings=True)
            combined_emb = torch.cat([o[0, last_t], context[0]], dim=-1).cpu().numpy()
            test_embeddings.append(np.append(combined_emb, int(rid)))

    pred_df = pd.DataFrame(pred_rows).sort_values("rally_uid")
    sample_df = pd.read_csv(args.sample)
    if len(sample_df) == 0: out = pred_df
    else: out = sample_df.drop(columns=["serverGetPoint","pointId","actionId"], errors="ignore").merge(pred_df, on="rally_uid", how="left")
    desired_cols = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out[desired_cols]
    out.to_csv(args.out, index=False)
    print(f"✅ 已成功儲存 Bi-LSTM 預測結果至: {args.out}")
    
    # 儲存 Test Embeddings
    test_emb_df = pd.DataFrame(test_embeddings)
    test_emb_df.rename(columns={test_emb_df.columns[-1]: "rally_uid"}, inplace=True)
    test_emb_df.to_csv("output/test_embeddings.csv", index=False)
    print("✅ 測試集 Embeddings 儲存完畢！(準備好給 XGBoost Stacking 使用了)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/train.csv")
    ap.add_argument("--test", default="dataset/test_new.csv")
    ap.add_argument("--sample", default="dataset/sample_submission.csv")
    ap.add_argument("--out", default="output/baseline_bilstm2.csv")
    ap.add_argument("--epochs", type=int, default=30) 
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=32) 
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_act", type=float, default=0.4)
    ap.add_argument("--w_pt", type=float, default=0.4)
    ap.add_argument("--w_rly", type=float, default=0.2)
    
    args = ap.parse_args()
    args.out = out_path_with_task_suffix(args.out, args.w_act, args.w_pt, args.w_rly)
    main(args)