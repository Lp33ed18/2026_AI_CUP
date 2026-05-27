# run_knn_with_lstm_point.py
"""
KNN (action, rally) + LSTM (point) hybrid baseline.
- actionId, serverGetPoint: KNN on sequence summaries (mean/max/min/std/last-first delta)
- pointId: use LSTM token-level point logits (last token argmax) from a MultiTask LSTM
Usage:
python run_knn_with_lstm_point.py --train train.csv --test test.csv --sample sample_submission.csv --out submission_hybrid.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PAD_TOKEN = 0

# -------------------------
# Reuse LSTM MultiTask model (from your run_lstm_final.py)
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
# Helpers (encoding, padding, seq summary)
# -------------------------
def normalize_scores(df, cap=11):
    df['scoreSelf'] = pd.to_numeric(df['scoreSelf'], errors='coerce').fillna(-1).astype(int)
    df['scoreOther'] = pd.to_numeric(df['scoreOther'], errors='coerce').fillna(-1).astype(int)
    df['is_deuce'] = ((df['scoreSelf'] >= 10) & (df['scoreOther'] >= 10)).astype(int)
    df['scoreSelf_capped'] = df['scoreSelf'].clip(upper=cap)
    df['scoreOther_capped'] = df['scoreOther'].clip(upper=cap)
    df['score_diff'] = (df['scoreSelf'] - df['scoreOther']).clip(-cap, cap)
    return df

def pad2d(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    out[:len(a)] = a
    return out

def pad2d_cap(a, m, pad_val=PAD_TOKEN):
    out = np.full((m, a.shape[1]), pad_val, dtype=np.int64)
    T = min(len(a), m)
    out[:T] = a[:T]
    return out, T

def build_cats(train, FEATURES):
    return {c: pd.Categorical(train[c]).categories for c in FEATURES}

def encode_frame_with_unk(df, FEATURES, cats):
    outs = []
    for col in FEATURES:
        dtype = pd.CategoricalDtype(categories=cats[col])
        s = pd.Series(df[col])
        codes = s.astype(dtype).cat.codes  # unseen -> -1
        unk_idx = len(cats[col])
        codes = codes.replace(-1, unk_idx)
        codes = codes + 1  # shift so 0 reserved for PAD
        outs.append(np.asarray(codes, dtype=np.int64))
    return np.stack(outs, axis=1)  # (T, F)

def seq_summary_features(enc_seq):
    # enc_seq: (T, F) integer codes
    arr = enc_seq.astype(float)
    mean = arr.mean(axis=0)
    mx   = arr.max(axis=0)
    mn   = arr.min(axis=0)
    std  = arr.std(axis=0)
    last = arr[-1,:]
    first = arr[0,:]
    delta = last - first
    feats = np.concatenate([mean, mx, mn, std, delta], axis=0)
    return feats

def build_rally_summaries(df, FEATURES, cats):
    out = {}
    for rid, g in df.groupby("rally_uid"):
        enc = encode_frame_with_unk(g, FEATURES, cats)
        if enc.shape[0] == 0:
            continue
        feat = seq_summary_features(enc)
        # labels for training KNN: use last frame's action/point and rally-level server label from first row
        action_label = int(g["actionId"].iloc[-1])
        point_label  = int(g["pointId"].iloc[-1])
        server_label = int(g["serverGetPoint"].iloc[0]) if "serverGetPoint" in g.columns else 0
        out[int(rid)] = (feat, action_label, point_label, server_label)
    return out

# -------------------------
# Main
# -------------------------
def main(args):
    os.makedirs("result", exist_ok=True)
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

    cats = build_cats(train, FEATURES)

    # -------------------------
    # Build KNN training data (sequence summaries)
    # -------------------------
    train_rallies = build_rally_summaries(train, FEATURES, cats)
    test_rallies  = build_rally_summaries(test, FEATURES, cats)

    train_ids = sorted(train_rallies.keys())
    if len(train_ids) == 0:
        raise RuntimeError("No training rallies found. Check train file and rally_uid grouping.")

    X_train = np.stack([train_rallies[r][0] for r in train_ids])
    yA_train = np.array([train_rallies[r][1] for r in train_ids])
    yP_train = np.array([train_rallies[r][2] for r in train_ids])
    yR_train = np.array([train_rallies[r][3] for r in train_ids])

    n_train_rallies = X_train.shape[0]
    print(f"[Hybrid] Number of training rally sequences used: {n_train_rallies}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(float))

    k = args.k
    neighbors_to_use = min(k, n_train_rallies)
    knn_action = KNeighborsClassifier(n_neighbors=neighbors_to_use, weights="distance")
    knn_server = KNeighborsClassifier(n_neighbors=neighbors_to_use, weights="distance")

    knn_action.fit(X_train_scaled, yA_train)
    knn_server.fit(X_train_scaled, yR_train)

    # -------------------------
    # Initialize LSTM for pointId (use same encoding scheme)
    # -------------------------
    # Build token vocab sizes per feature (num_tokens_per_feature)
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]
    # Determine point/action classes from training data (unique values)
    act_classes = np.sort(train["actionId"].unique()); n_act = len(act_classes)
    pt_classes  = np.sort(train["pointId"].unique());  n_pt  = len(pt_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # instantiate MultiTaskLSTM with same hyperparams as your LSTM baseline (defaults)
    model = MultiTaskLSTM(num_tokens_per_feature, n_act, n_pt,
                         emb_dim=args.emb, hidden=args.hidden,
                         num_layers=max(1,args.layers), dropout=args.drop,
                         bidirectional=args.bidirectional).to(device)

    # If you have pretrained LSTM weights, load them here:
    if args.lstm_ckpt and os.path.exists(args.lstm_ckpt):
        ckpt = torch.load(args.lstm_ckpt, map_location=device)
        model.load_state_dict(ckpt)
        print(f"[Hybrid] Loaded LSTM checkpoint from {args.lstm_ckpt}")
    else:
        print("[Hybrid] No LSTM checkpoint provided; LSTM will be used with random init (not recommended).")

    model.eval()

    # -------------------------
    # Inference: for each test rally
    # - actionId, serverGetPoint from KNN (sequence summary)
    # - pointId from LSTM token-level last token argmax
    # -------------------------
    pred_rows = []
    info_rows = []

    with torch.no_grad():
        for rid, g in test.groupby("rally_uid"):
            # KNN features: sequence summary of the rally (we already built test_rallies)
            if int(rid) in test_rallies:
                feat, _, _, _ = test_rallies[int(rid)]
                Xs = feat.reshape(1, -1).astype(float)
                Xs_scaled = scaler.transform(Xs)
                neighbors_used = min(k, n_train_rallies)
                a_pred = int(knn_action.predict(Xs_scaled)[0])
                # server probability via predict_proba if available
                if hasattr(knn_server, "predict_proba"):
                    probs = knn_server.predict_proba(Xs_scaled)
                    prob1 = 0.0
                    if 1 in knn_server.classes_:
                        prob1 = float(probs[0, list(knn_server.classes_).index(1)])
                    else:
                        prob1 = float(yR_train.mean())
                else:
                    prob1 = float(knn_server.predict(Xs_scaled)[0])
            else:
                # fallback if test rally not encoded (shouldn't happen)
                a_pred = int(act_classes[0])
                prob1 = float(yR_train.mean())
                neighbors_used = 0

            # LSTM point prediction: encode full rally, pad, run model, take last token point logits
            Xg = encode_frame_with_unk(g, FEATURES, cats)
            Xp, T = pad2d_cap(Xg, min(max(1, Xg.shape[0]), args.max_len))
            X_t = torch.tensor(Xp[None,...], dtype=torch.long, device=device)
            L_t = torch.tensor([max(1, T)], dtype=torch.long, device=device)
            la, lp, lr = model(X_t, L_t)
            last_t = L_t.item() - 1
            p_idx = int(torch.argmax(lp[0, last_t]).item())
            point_pred = int(pt_classes[p_idx]) if p_idx < len(pt_classes) else int(pt_classes[-1])

            pred_rows.append({"rally_uid": int(rid), "actionId": a_pred, "pointId": point_pred, "serverGetPoint": prob1})
            info_rows.append({"rally_uid": int(rid), "neighbors_used": neighbors_used, "train_rallies_used": n_train_rallies})

    pred_df = pd.DataFrame(pred_rows)
    info_df = pd.DataFrame(info_rows)
    info_df.to_csv("result/knn_info.csv", index=False)
    print("[Hybrid] Wrote neighbor info to result/knn_info.csv")

    # Merge with sample like LSTM script
    sub_template = sample
    if len(sub_template) > 0 and 'rally_uid' in sub_template.columns:
        out = sub_template.drop(columns=["serverGetPoint","pointId","actionId"], errors="ignore").merge(pred_df, on="rally_uid", how="left")
    else:
        out = pred_df

    cols_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out.reindex(columns=[c for c in cols_order if c in out.columns])
    out.to_csv(args.out, index=False)
    print(f"[Hybrid] Submission written to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="inputs/train.csv")
    ap.add_argument("--test", default="inputs/test_new.csv")
    ap.add_argument("--sample", default="result/sample_submission.csv")
    ap.add_argument("--out", default="result/submission_hybrid.csv")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--cap", type=int, default=11)
    ap.add_argument("--bidirectional", type=bool, default=True)
    ap.add_argument("--lstm_ckpt", type=str, default="", help="optional path to pretrained LSTM state_dict")
    args = ap.parse_args()
    main(args)
