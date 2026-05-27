# run_stgnn_gat.py
"""
STGNN training & inference script with positional embedding and GATConv for AI CUP 2026 Table Tennis task.

Features:
- Learnable positional embedding (indexed by strikeNumber)
- GATConv layers with residual connections and LayerNorm
- Mixed-precision training (torch.amp) + Cosine LR scheduler
- Debug checks A/B/C available with --debug
- Corrected class mapping: labels are mapped to contiguous indices at dataset creation
- Outputs CSV in baseline order: ["rally_uid","actionId","pointId","serverGetPoint"]
"""

import argparse
import math
import random
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.metrics import f1_score, roc_auc_score
import sys
import os

# PyG imports (compatible import with loader)
try:
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
except Exception as e:
    try:
        from torch_geometric.data import Data, Dataset, DataLoader as PyGDataLoader
        from torch_geometric.nn import GATConv, global_mean_pool
    except Exception:
        raise ImportError("This script requires torch_geometric. Install with 'pip install torch-geometric' following the official instructions.") from e

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

PAD_TOKEN = 0

# -------------------------
# Helper: padding functions (copied/adapted from baseline)
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
# Score normalization (copied/adapted)
# -------------------------
def normalize_scores(df, cap=11):
    df['scoreSelf'] = pd.to_numeric(df.get('scoreSelf', pd.Series([-1]*len(df))), errors='coerce').fillna(-1).astype(int)
    df['scoreOther'] = pd.to_numeric(df.get('scoreOther', pd.Series([-1]*len(df))), errors='coerce').fillna(-1).astype(int)
    df['is_deuce'] = ((df['scoreSelf'] >= 10) & (df['scoreOther'] >= 10)).astype(int)
    df['scoreSelf_capped'] = df['scoreSelf'].clip(upper=cap)
    df['scoreOther_capped'] = df['scoreOther'].clip(upper=cap)
    df['score_diff'] = (df['scoreSelf'] - df['scoreOther']).clip(-cap, cap)
    return df

# -------------------------
# Encode frames with UNK handling (adapted from baseline)
# -------------------------
def build_categorical_maps(df, FEATURES):
    cats = {c: pd.Categorical(df[c]).categories for c in FEATURES}
    return cats

def encode_frame_with_unk_from_cats(df, FEATURES, cats):
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

# -------------------------
# PyG Dataset: one Data per rally
#   Now accepts act_id2idx / pt_id2idx to map labels to contiguous indices
# -------------------------
class RallyGraphDataset(Dataset):
    def __init__(self, df, FEATURES, cats, max_len=None, is_train=True, act_id2idx=None, pt_id2idx=None):
        super().__init__()
        self.groups = list(df.groupby("rally_uid"))
        self.FEATURES = FEATURES
        self.cats = cats
        self.max_len = max_len
        self.is_train = is_train
        # mapping dicts: original id -> index (0..N-1)
        self.act_id2idx = act_id2idx
        self.pt_id2idx = pt_id2idx

    def len(self):
        return len(self.groups)

    def get(self, idx):
        rid, g = self.groups[idx]
        g = g.sort_values("strikeNumber")
        # require at least 2 strokes to form input -> label
        if len(g) < 2:
            # create a minimal dummy graph with one padded node
            X = np.zeros((1, len(self.FEATURES)), dtype=np.int64)
            edge_index = torch.empty((2,0), dtype=torch.long)
            data = Data(x=torch.tensor(X, dtype=torch.long),
                        edge_index=edge_index,
                        rally_uid=torch.tensor([int(rid)], dtype=torch.long),
                        last_index=torch.tensor([0], dtype=torch.long))
            # labels if available
            if self.is_train:
                # map to indices if mapping provided, else keep original id (fallback)
                a_raw = int(g["actionId"].iloc[-1]) if "actionId" in g.columns else -1
                p_raw = int(g["pointId"].iloc[-1]) if "pointId" in g.columns else -1
                a_idx = self.act_id2idx.get(a_raw, -1) if self.act_id2idx is not None else a_raw
                p_idx = self.pt_id2idx.get(p_raw, -1) if self.pt_id2idx is not None else p_raw
                data.y_strike = torch.tensor([int(a_idx)], dtype=torch.long)
                data.y_point  = torch.tensor([int(p_idx)], dtype=torch.long)
                data.y_result = torch.tensor([int(g["serverGetPoint"].iloc[0])], dtype=torch.long)
            return data

        X_full = encode_frame_with_unk_from_cats(g, self.FEATURES, self.cats)  # shape [T, F]
        # input is first n-1 strokes
        X_in = X_full[:-1]
        T = len(X_in)
        # build time edges (bidirectional)
        if T >= 2:
            src = np.arange(0, T-1, dtype=np.int64)
            dst = np.arange(1, T, dtype=np.int64)
            edge_index = np.concatenate([np.stack([src, dst], axis=0), np.stack([dst, src], axis=0)], axis=1)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)

        x = torch.tensor(X_in, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.rally_uid = torch.tensor([int(rid)], dtype=torch.long)
        data.last_index = torch.tensor([T-1], dtype=torch.long)  # index of last node in this graph (0-based)

        if self.is_train:
            # map labels to contiguous indices using provided dicts
            a_raw = int(g["actionId"].iloc[-1]) if "actionId" in g.columns else -1
            p_raw = int(g["pointId"].iloc[-1]) if "pointId" in g.columns else -1
            a_idx = self.act_id2idx.get(a_raw, -1) if self.act_id2idx is not None else a_raw
            p_idx = self.pt_id2idx.get(p_raw, -1) if self.pt_id2idx is not None else p_raw
            data.y_strike = torch.tensor(int(a_idx), dtype=torch.long)
            data.y_point  = torch.tensor(int(p_idx), dtype=torch.long)
            data.y_result = torch.tensor(int(g["serverGetPoint"].iloc[0]) if "serverGetPoint" in g.columns else 0, dtype=torch.long)

        return data

# -------------------------
# STGNN Model (Per-step GAT + GRU fusion) with positional embedding
# -------------------------
class StrokeEncoder(nn.Module):
    def __init__(self, num_tokens_per_feature, emb_dim=32, node_hidden=128, max_len=512, pos_emb_dim=None):
        super().__init__()
        self.num_features = len(num_tokens_per_feature)
        self.emb_dim = emb_dim
        self.embs = nn.ModuleList([nn.Embedding(n + 1, emb_dim, padding_idx=PAD_TOKEN) for n in num_tokens_per_feature])
        self.proj = nn.Linear(self.num_features * emb_dim, node_hidden)
        # positional embedding: learnable, size = max_len+1, dim = node_hidden (or pos_emb_dim if provided)
        self.max_len = max_len
        pos_dim = pos_emb_dim if pos_emb_dim is not None else node_hidden
        self.pos_emb = nn.Embedding(self.max_len + 1, pos_dim)
        # if pos_dim != node_hidden, project to node_hidden
        if pos_dim != node_hidden:
            self.pos_proj = nn.Linear(pos_dim, node_hidden)
        else:
            self.pos_proj = None

    def forward(self, x):
        # x: [N, F] long, last column expected to be strikeNumber (encoded)
        feats = []
        for i, emb in enumerate(self.embs):
            feats.append(emb(x[:, i]))
        h = torch.cat(feats, dim=-1)  # [N, F*emb_dim]
        h = self.proj(h)             # [N, node_hidden]

        # positional index: use last column (strikeNumber). clamp to [0, max_len]
        pos_idx = x[:, -1].clamp(0, self.max_len).long()
        pos_v = self.pos_emb(pos_idx)  # [N, pos_dim]
        if self.pos_proj is not None:
            pos_v = self.pos_proj(pos_v)  # [N, node_hidden]

        # combine: add positional vector to node representation
        h = h + pos_v
        return h  # [N, node_hidden]

class PerStepGAT(nn.Module):
    def __init__(self, node_in_dim, gnn_hidden=128, num_layers=2, heads=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_projs = nn.ModuleList()
        self.heads = heads
        # first layer: map node_in_dim -> gnn_hidden
        out_per_head = gnn_hidden // heads
        if out_per_head * heads != gnn_hidden:
            # ensure divisible
            out_per_head = max(1, gnn_hidden // heads)
            gnn_hidden = out_per_head * heads
        self.gnn_hidden = gnn_hidden

        self.layers.append(GATConv(node_in_dim, out_per_head, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(gnn_hidden))
        # residual projection if dims mismatch
        if node_in_dim != gnn_hidden:
            self.res_projs.append(nn.Linear(node_in_dim, gnn_hidden))
        else:
            self.res_projs.append(nn.Identity())

        for _ in range(num_layers - 1):
            self.layers.append(GATConv(gnn_hidden, out_per_head, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(gnn_hidden))
            self.res_projs.append(nn.Identity())

    def forward(self, x, edge_index):
        # x: [N, node_in_dim]
        for conv, ln, rproj in zip(self.layers, self.norms, self.res_projs):
            h = conv(x, edge_index)  # returns [N, out_per_head * heads] == gnn_hidden
            h = F.elu(h)
            h = ln(h)
            # residual (project x if needed)
            res = rproj(x)
            x = h + res
        return x  # [N, gnn_hidden]

class STGNNModel(nn.Module):
    def __init__(self, num_tokens_per_feature, n_act, n_pt,
                 emb_dim=32, node_hidden=128, gnn_hidden=128, gnn_layers=2, gnn_heads=4,
                 rnn_hidden=128, rnn_layers=1, dropout=0.2, max_len=512, gat_dropout=0.0):
        super().__init__()
        # pass max_len to encoder for positional embedding
        self.encoder = StrokeEncoder(num_tokens_per_feature, emb_dim=emb_dim, node_hidden=node_hidden, max_len=max_len)
        self.gnn = PerStepGAT(node_in_dim=node_hidden, gnn_hidden=gnn_hidden, num_layers=gnn_layers, heads=gnn_heads, dropout=gat_dropout)
        self.rnn = nn.GRU(input_size=gnn_hidden, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(dropout)

        # heads
        self.head_strike = nn.Linear(rnn_hidden, n_act)  # token-level classes (we will use last token)
        self.head_point  = nn.Linear(rnn_hidden, n_pt)
        self.head_result = nn.Linear(rnn_hidden, 1)

    def forward(self, batch):
        # batch is a PyG Batch
        x_enc = self.encoder(batch.x)  # [N, node_hidden]
        x_g = self.gnn(x_enc, batch.edge_index)  # [N, gnn_hidden]

        batch_idx = batch.batch  # [N]
        num_graphs = int(batch_idx.max().item()) + 1
        seqs = []
        lengths = []
        for gid in range(num_graphs):
            mask = (batch_idx == gid)
            seq = x_g[mask]  # [T_g, gnn_hidden]
            lengths.append(seq.size(0))
            seqs.append(seq)

        max_len = max(lengths)
        padded = []
        for seq in seqs:
            T = seq.size(0)
            if T < max_len:
                pad = torch.zeros((max_len - T, seq.size(1)), device=seq.device, dtype=seq.dtype)
                seq_p = torch.cat([seq, pad], dim=0)
            else:
                seq_p = seq
            padded.append(seq_p.unsqueeze(0))
        seq_batch = torch.cat(padded, dim=0)  # [B, max_len, gnn_hidden]

        out, h = self.rnn(seq_batch)  # out: [B, max_len, rnn_hidden]
        out = self.drop(out)

        last_idxs = [l - 1 for l in lengths]
        last_tensor = torch.tensor(last_idxs, dtype=torch.long, device=out.device)
        batch_range = torch.arange(out.size(0), device=out.device)
        last_h = out[batch_range, last_tensor, :]  # [B, rnn_hidden]

        logit_strike = self.head_strike(last_h)
        logit_point  = self.head_point(last_h)
        logit_result = self.head_result(last_h)

        return logit_strike, logit_point, logit_result

# -------------------------
# Training & evaluation utilities (mixed-precision aware)
# -------------------------
def train_one_epoch(model, loader, optimizer, device, ce_action, ce_point, bce_rally, weights, scaler):
    model.train()
    total_loss = 0.0
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            la, lp, lr = model(batch)
            yA = batch.y_strike.to(device)
            yP = batch.y_point.to(device)
            yR = batch.y_result.to(device).float()

            loss_s = ce_action(la, yA)
            loss_p = ce_point(lp, yP)
            lr_logits = lr.squeeze(1)
            loss_r = bce_rally(lr_logits, yR)
            loss = weights[0]*loss_s + weights[1]*loss_p + weights[2]*loss_r

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * yA.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, ce_action, ce_point, bce_rally):
    model.eval()
    total_loss = 0.0
    allA, allAp, allP, allPp, allR, allRp = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            la, lp, lr = model(batch)
            yA = batch.y_strike.to(device)
            yP = batch.y_point.to(device)
            yR = batch.y_result.to(device).float()

            loss_s = ce_action(la, yA)
            loss_p = ce_point(lp, yP)
            lr_logits = lr.squeeze(1)
            loss_r = bce_rally(lr_logits, yR)
            loss = 0.4*loss_s + 0.4*loss_p + 0.2*loss_r
            total_loss += loss.item() * yA.size(0)

            probs_r = torch.sigmoid(lr_logits).detach().cpu().tolist()
            predA = la.argmax(-1).detach().cpu().numpy()
            predP = lp.argmax(-1).detach().cpu().numpy()
            yA_np = yA.detach().cpu().numpy()
            yP_np = yP.detach().cpu().numpy()

            allR += yR.detach().cpu().tolist(); allRp += probs_r
            maskA = (yA_np != -1)
            maskP = (yP_np != -1)
            allA += yA_np[maskA].tolist(); allAp += predA[maskA].tolist()
            allP += yP_np[maskP].tolist(); allPp += predP[maskP].tolist()

    try:
        f1A = f1_score(allA, allAp, average="macro") if len(allA) else 0.0
        f1P = f1_score(allP, allPp, average="macro") if len(allP) else 0.0
        auc = roc_auc_score(allR, allRp) if len(set(allR))>1 else 0.5
    except Exception:
        f1A, f1P, auc = 0.0, 0.0, 0.5
    final = 0.4*f1A + 0.4*f1P + 0.2*auc
    return total_loss / len(loader.dataset), f1A, f1P, auc, final

# -------------------------
# Debug checks (A/B/C)
# -------------------------
def debug_check_A_label_source(train_df, FEATURES, cats, n_samples=10):
    print("=== Debug Check A: label source verification (show last stroke labels and X_in) ===")
    for i, (rid, g) in enumerate(train_df.groupby("rally_uid")):
        if i >= n_samples: break
        g = g.sort_values("strikeNumber")
        last_strikeId = int(g['strikeId'].iloc[-1]) if 'strikeId' in g.columns else None
        last_actionId = int(g['actionId'].iloc[-1]) if 'actionId' in g.columns else None
        last_pointId  = int(g['pointId'].iloc[-1])  if 'pointId' in g.columns else None
        print(f"RID {rid} len {len(g)} last_strikeId={last_strikeId} last_actionId={last_actionId} last_pointId={last_pointId}")
        X_full = encode_frame_with_unk_from_cats(g, FEATURES, cats)
        X_in = X_full[:-1] if X_full.shape[0] > 1 else X_full[:1]
        print("  X_in shape", X_in.shape, "first row", X_in[0] if X_in.shape[0]>0 else None)
    print("=== End Check A ===\n")

def debug_check_B_off_by_one(train_df, FEATURES, cats, n_samples=10):
    print("=== Debug Check B: off-by-one leakage detection ===")
    for i, (rid, g) in enumerate(train_df.groupby("rally_uid")):
        if i >= n_samples: break
        g = g.sort_values("strikeNumber")
        X_full = encode_frame_with_unk_from_cats(g, FEATURES, cats)
        if X_full.shape[0] < 2:
            print(f"RID {rid} too short (T={X_full.shape[0]}), skipping")
            continue
        X_in = X_full[:-1]
        label_action = int(g['actionId'].iloc[-1]) if 'actionId' in g.columns else None
        label_point  = int(g['pointId'].iloc[-1])  if 'pointId' in g.columns else None
        last_row = X_in[-1]
        print(f"RID {rid} label_action={label_action} label_point={label_point} X_in[-1]={last_row}")
    print("=== End Check B ===\n")

def debug_check_C_val_distribution(full_dataset, val_ds, n_classes_action=None, n_classes_point=None):
    print("=== Debug Check C: validation set class distributions ===")
    try:
        val_indices = val_ds.indices
    except Exception:
        val_indices = list(range(len(full_dataset)))[-len(val_ds):]
    yA = []
    yP = []
    for idx in val_indices:
        data = full_dataset[idx]
        yA.append(int(data.y_strike))
        yP.append(int(data.y_point))
    cntA = Counter(yA)
    cntP = Counter(yP)
    print("Validation action distribution (value:count):", dict(cntA))
    print("Validation point distribution (value:count):", dict(cntP))
    print("Total val samples:", len(val_indices))
    print("=== End Check C ===\n")

# -------------------------
# Main: data prep, model, train, inference
# -------------------------
def main(args):
    # logging to file
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    log_file = open(os.path.join(os.path.dirname(args.out), "stgnn_log_gat.txt"), "w", encoding="utf-8")
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
    sys.stdout = Tee(sys.stdout, log_file)

    train_df = pd.read_csv(args.train)
    test_df  = pd.read_csv(args.test)
    sample   = pd.read_csv(args.sample) if args.sample else pd.DataFrame()

    train_df = normalize_scores(train_df, cap=args.cap)
    test_df  = normalize_scores(test_df, cap=args.cap)

    # FEATURES used for node encoding (must match encode/embedding order)
    FEATURES = [
        "sex","handId","strengthId","spinId",
        "pointId","actionId","positionId","strikeId",
        "scoreSelf_capped","scoreOther_capped","score_diff","is_deuce","strikeNumber"
    ]

    # clip strikeNumber to avoid extremely long sequences
    train_df["strikeNumber"] = train_df["strikeNumber"].clip(0, args.max_len-1)
    test_df["strikeNumber"]  = test_df["strikeNumber"].clip(0, args.max_len-1)

    # build categorical maps from training set
    cats = build_categorical_maps(train_df, FEATURES)

    # -------------------------
    # Derive class lists and mappings from training set (preserve appearance order)
    # Use dropna().unique() to keep original order of appearance
    # -------------------------
    act_classes = train_df["actionId"].dropna().unique()
    pt_classes  = train_df["pointId"].dropna().unique()
    # ensure numpy arrays
    act_classes = np.asarray(act_classes, dtype=np.int64)
    pt_classes = np.asarray(pt_classes, dtype=np.int64)
    n_act = len(act_classes)
    n_pt  = len(pt_classes)

    # mappings: original id -> index (0..N-1)
    act_id2idx = {int(v): i for i, v in enumerate(act_classes)}
    pt_id2idx  = {int(v): i for i, v in enumerate(pt_classes)}

    # num tokens per feature for embeddings (len(categories)+1 for UNK)
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]

    # Build train dataset (PyG) AFTER we have mappings so dataset stores indices
    full_dataset = RallyGraphDataset(train_df, FEATURES, cats, max_len=args.max_len, is_train=True,
                                     act_id2idx=act_id2idx, pt_id2idx=pt_id2idx)

    # split train/val
    n = len(full_dataset)
    if n == 0:
        raise RuntimeError("No training sequences constructed. Check train data and grouping by rally_uid.")
    val_size = int(n * args.val_size)
    train_size = n - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = PyGDataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = PyGDataLoader(val_ds, batch_size=max(args.batch*2, 64), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = STGNNModel(num_tokens_per_feature,
                       n_act=n_act, n_pt=n_pt,
                       emb_dim=args.emb, node_hidden=args.node_hidden,
                       gnn_hidden=args.gnn_hidden, gnn_layers=args.gnn_layers, gnn_heads=args.gnn_heads,
                       rnn_hidden=args.rnn_hidden, rnn_layers=args.rnn_layers,
                       dropout=args.drop, max_len=args.max_len, gat_dropout=args.gat_drop).to(device)

    # class weights for action/point from training set (to mimic baseline weighting)
    act_counts = np.bincount([act_id2idx[int(v)] for v in train_df["actionId"].dropna().astype(int).values], minlength=n_act) + 1
    pt_counts  = np.bincount([pt_id2idx[int(v)] for v in train_df["pointId"].dropna().astype(int).values], minlength=n_pt) + 1
    act_w = torch.tensor(1.0/act_counts, dtype=torch.float32); act_w = (act_w * (n_act/act_w.sum()))
    pt_w  = torch.tensor(1.0/pt_counts, dtype=torch.float32); pt_w  = (pt_w * (n_pt/pt_w.sum()))

    ce_action = nn.CrossEntropyLoss(weight=act_w.to(device))
    ce_point  = nn.CrossEntropyLoss(weight=pt_w.to(device))
    bce_rally = nn.BCEWithLogitsLoss()

    # Use AdamW and weight decay for better generalization
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed precision scaler (torch.amp recommended)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Scheduler: CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    weights = (args.w_s, args.w_p, args.w_r)

    # If debug flag set, run checks A/B/C before training
    if args.debug:
        print("\n*** RUNNING DEBUG CHECKS (A/B/C) BEFORE TRAINING ***\n")
        debug_check_A_label_source(train_df, FEATURES, cats, n_samples=10)
        debug_check_B_off_by_one(train_df, FEATURES, cats, n_samples=10)
        debug_check_C_val_distribution(full_dataset, val_ds, n_classes_action=n_act, n_classes_point=n_pt)
        print("\n*** END DEBUG CHECKS ***\n")

    # training loop with mixed precision and scheduler; save best model by 'final' metric
    best_final = -1.0
    best_epoch = -1
    model_dir = os.path.dirname(args.out) or "."
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, "best_stgnn_gat.pth")

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, ce_action, ce_point, bce_rally, weights, scaler)
        va_loss, f1A, f1P, auc, final = evaluate(model, val_loader, device, ce_action, ce_point, bce_rally)
        print(f"[Epoch {ep}/{args.epochs}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} F1_action={f1A:.4f} F1_point={f1P:.4f} AUC={auc:.4f} Final~{final:.4f}")

        # scheduler step (per epoch)
        try:
            scheduler.step()
        except Exception:
            pass

        # save best model
        if final > best_final:
            best_final = final
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "final": final
            }, best_model_path)
            print(f"Saved best model (epoch {ep}) to {best_model_path}")

    print(f"Training finished. Best final={best_final:.4f} at epoch {best_epoch}")

    # load best model for inference if available
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model from {best_model_path} for inference (final={ckpt.get('final', -1):.4f})")

    # Optional debug prints before inference (sample test rallies)
    if args.debug:
        print("\nDEBUG: sample test rallies (first 5) shapes and first rows:")
        for i, (rid, g) in enumerate(test_df.groupby("rally_uid")):
            if i >= 5: break
            g = g.sort_values("strikeNumber")
            Xg = encode_frame_with_unk_from_cats(g, FEATURES, cats)
            print(f"RID {rid} T_full {Xg.shape[0]} first_row {Xg[0] if Xg.shape[0]>0 else None}")
            X_in = Xg[:-1] if Xg.shape[0] > 1 else Xg[:1]
            print(f"  input_len {X_in.shape[0]} first_rows {X_in[:3]}")
        print("End sample test debug\n")

    # -------------------------
    # Inference on test set (use best model)
    # -------------------------
    pred_rows = []
    model.eval()
    with torch.no_grad():
        for rid, g in test_df.groupby("rally_uid"):
            g = g.sort_values("strikeNumber")
            Xg = encode_frame_with_unk_from_cats(g, FEATURES, cats)  # shape [T_full, F]
            if Xg.shape[0] < 2:
                X_in = Xg[:1]
            else:
                X_in = Xg[:-1]

            real_T = X_in.shape[0]
            if real_T > args.max_len:
                X_in = X_in[:args.max_len]
                real_T = args.max_len

            if real_T < args.max_len:
                pad_rows = np.zeros((args.max_len - real_T, X_in.shape[1]), dtype=np.int64)
                Xp = np.concatenate([X_in, pad_rows], axis=0)
            else:
                Xp = X_in

            x_tensor = torch.tensor(Xp, dtype=torch.long, device=device)

            if real_T >= 2:
                src = np.arange(0, real_T-1, dtype=np.int64)
                dst = np.arange(1, real_T, dtype=np.int64)
                edge_index = np.concatenate([np.stack([src, dst], axis=0), np.stack([dst, src], axis=0)], axis=1)
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
            else:
                edge_index = torch.empty((2,0), dtype=torch.long, device=device)

            data = Data(x=x_tensor, edge_index=edge_index)
            data.batch = torch.zeros(x_tensor.size(0), dtype=torch.long, device=device)
            batch = data.to(device)

            la, lp, lr = model(batch)
            a_idx = int(torch.argmax(la, dim=-1).item()) if la.numel() else 0
            p_idx = int(torch.argmax(lp, dim=-1).item()) if lp.numel() else 0
            lr_val = lr.squeeze(1) if lr.dim() == 2 else lr
            s_prob = float(torch.sigmoid(lr_val).item()) if lr_val.numel() else 0.0
            # map back to original class ids using act_classes / pt_classes
            action_pred = int(act_classes[a_idx]) if a_idx < len(act_classes) else int(act_classes[-1])
            point_pred  = int(pt_classes[p_idx])  if p_idx < len(pt_classes)  else int(pt_classes[-1])
            pred_rows.append({"rally_uid": int(rid), "actionId": action_pred, "pointId": point_pred, "serverGetPoint": s_prob})

    pred_df = pd.DataFrame(pred_rows)

    # merge with sample template if provided (baseline behavior)
    sub_template = sample
    if len(sub_template) > 0 and 'rally_uid' in sub_template.columns:
        out = sub_template.drop(columns=["serverGetPoint","pointId","actionId"], errors="ignore").merge(pred_df, on="rally_uid", how="left")
    else:
        out = pred_df

    # ensure column order as requested
    cols_order = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
    out = out.reindex(columns=[c for c in cols_order if c in out.columns])

    out.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="inputs/train.csv")
    ap.add_argument("--test", default="inputs/test_new.csv")
    ap.add_argument("--sample", default="result/sample_submission.csv")
    ap.add_argument("--out", default="result/submission.csv")
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--node_hidden", type=int, default=128)
    ap.add_argument("--gnn_hidden", type=int, default=128)
    ap.add_argument("--gnn_layers", type=int, default=2)
    ap.add_argument("--gnn_heads", type=int, default=4)
    ap.add_argument("--gat_drop", type=float, default=0.0)
    ap.add_argument("--rnn_hidden", type=int, default=128)
    ap.add_argument("--rnn_layers", type=int, default=1)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--cap", type=int, default=11)
    ap.add_argument("--w_s", type=float, default=0.375)  # weight for action loss (to mirror baseline)
    ap.add_argument("--w_p", type=float, default=0.5)    # weight for point loss
    ap.add_argument("--w_r", type=float, default=0.125)  # weight for rally loss
    ap.add_argument("--debug", action="store_true", help="Run debug checks A/B/C before training and print sample test inputs")
    args = ap.parse_args()
    main(args)
