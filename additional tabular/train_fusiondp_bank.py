#!/usr/bin/env python3
import argparse
import json
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# ------------- imports ------------------------------------------
from torch.func import functional_call, vmap
# -----------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
# Bank Marketing schema (bank-additional / bank-additional-full)
# ============================================================
# Numeric features (common in bank-additional dataset)
BANK_NUMERIC = [
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

BANK_CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "month",
    "poutcome",
]

# ============================================================
# Repro
# ============================================================
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# ============================================================
# MLP (Always returns (logits, rep))
# ============================================================
class MLP(nn.Module):
    def __init__(self, d_in: int, h1: int = 64, h2: int = 32, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.out = nn.Linear(h2, 1)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(self.act(self.ln1(self.fc1(x))))
        rep = self.drop(self.act(self.ln2(self.fc2(h))))
        logits = self.out(rep).squeeze(-1)
        return logits, rep

    def get_rep(self, x):
        with torch.no_grad():
            _, rep = self.forward(x)
            return rep

# ============================================================
# Torch dataset (zero-copy from numpy)
# ============================================================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # expect contiguous float32
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32)).view(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    ys, ps = [], []
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        logits, _ = model(Xb)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(yb.numpy())
        ps.append(prob)
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    return acc, auroc, aupr, f1

# ============================================================
# DP utils
# ============================================================
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    epochs: int = 10,
    max_sigma: float = 1e6,
) -> float:
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")

    def eps_for_sigma(sigma: float) -> float:
        steps = int(np.ceil(epochs / sample_rate))
        orders = [1 + x / 10.0 for x in range(1, 1000)] + list(range(101, 1024))
        rdp = compute_rdp(q=sample_rate, noise_multiplier=sigma, steps=steps, orders=orders)
        eps, _ = get_privacy_spent(orders=orders, rdp=rdp, delta=target_delta)
        return float(eps)

    lo, hi = 1e-5, 50.0
    while eps_for_sigma(hi) > target_epsilon:
        hi *= 2.0
        if hi > max_sigma:
            return max_sigma

    for _ in range(60):
        mid = (lo + hi) / 2.0
        e = eps_for_sigma(mid)
        if e > target_epsilon:
            lo = mid
        else:
            hi = mid
        if abs(e - target_epsilon) <= epsilon_tolerance:
            break
    return hi

def project_to_W(params: List[torch.Tensor], M: float = 10.0):
    with torch.no_grad():
        for p in params:
            norm = p.norm()
            if norm > M:
                p.mul_(M / (norm + 1e-12))

def dp_sgd_update(model, batch, batch_labels, loss_fn, max_grad_norm, noise_multiplier):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    B = batch.size(0)

    def _loss(p, b, x, y):
        logits, _ = functional_call(model, (p, b), (x.unsqueeze(0),))
        return loss_fn(logits.squeeze(), y)

    per_ex_grads = vmap(
        torch.func.grad(_loss),
        in_dims=(None, None, 0, 0),
        randomness="different",
    )(params, buffers, batch, batch_labels.float())

    per_sample_norms_sq = torch.stack([
        g.view(B, -1).pow(2).sum(dim=1) for g in per_ex_grads.values()
    ]).sum(dim=0)
    per_sample_norms = torch.sqrt(per_sample_norms_sq)
    clip_factors = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

    aggregated_grads = []
    for name, _ in model.named_parameters():
        g_batch = per_ex_grads[name]
        view_shape = [B] + [1] * (g_batch.ndim - 1)
        factor = clip_factors.view(*view_shape)
        g_sum = (g_batch * factor).sum(dim=0)
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * max_grad_norm,
            size=g_sum.shape,
            device=g_sum.device,
        )
        aggregated_grads.append(g_sum / B + noise / B)

    return aggregated_grads

def feature_dp_update_sample(
    model,
    X,
    X_pub,
    Y,
    loss_fn,
    max_grad_norm,
    noise_multiplier,
    alpha=1.0,
    M=10.0,
    batch_priv_size=128,
    batch_pub_size=4096,
):
    N = X.shape[0]
    p_poisson = batch_priv_size / max(N, 1)
    mask = (torch.rand(N, device=DEVICE) < p_poisson)

    B_priv = X[mask]
    B_priv_pub = X_pub[mask]
    B_priv_labels = Y[mask]

    k = min(batch_pub_size, N)
    perm = torch.randperm(N, device=DEVICE)[:k]
    B_pub_pub = X_pub[perm]
    B_pub_labels = Y[perm]

    # Public gradient
    model.zero_grad()
    logits_pub, _ = model(B_pub_pub)
    loss_pub = loss_fn(logits_pub.squeeze(), B_pub_labels).mean()
    loss_pub.backward()
    grad_pub = [param.grad.detach().clone() for param in model.parameters()]
    model.zero_grad()

    if B_priv.size(0) == 0:
        return grad_pub

    def _loss_fn(params, buffers, x, x_pub, y):
        logits_total, _ = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        logits_pub, _ = functional_call(model, (params, buffers), (x_pub.unsqueeze(0),))
        return loss_fn(logits_total.squeeze(), y) - loss_fn(logits_pub.squeeze(), y)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    grads = vmap(
        torch.func.grad(_loss_fn),
        in_dims=(None, None, 0, 0, 0),
        randomness="different",
    )(params, buffers, B_priv, B_priv_pub, B_priv_labels)

    B = B_priv.size(0)
    per_sample_norms_sq = torch.stack([
        g.view(B, -1).pow(2).sum(dim=1) for g in grads.values()
    ]).sum(dim=0)
    per_sample_norms = torch.sqrt(per_sample_norms_sq)
    clip_factors = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

    aggregated_grads = []
    for name, _ in model.named_parameters():
        g_batch = grads[name]
        view_shape = [B] + [1] * (g_batch.ndim - 1)
        factor = clip_factors.view(*view_shape)
        g_sum = (g_batch * factor).sum(dim=0)
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * max_grad_norm,
            size=g_sum.shape,
            device=g_sum.device,
        )
        aggregated_grads.append(g_sum / B + noise / B)

    total_grads = [grad_pub[i] + alpha * aggregated_grads[i] for i in range(len(aggregated_grads))]
    return total_grads

def rep_consistency_loss(h_real, h_imp, C_h, beta):
    delta = h_real - h_imp
    norms = delta.norm(p=2, dim=1, keepdim=True)
    scale = torch.clamp(C_h / (norms + 1e-12), max=1.0)
    delta_clip = delta * scale
    corr_i = delta_clip.pow(2).sum(dim=1)
    corr_i = corr_i / (C_h ** 2)
    return beta * corr_i

def calibrate4(
    model,
    X,
    X_pub,
    Y,
    loss_fn,
    max_grad_norm,
    noise_multiplier,
    C_h,
    beta,
    alpha,
    batch_priv_size=128,
    batch_pub_size=4096,
    device="cuda",
):
    N = X.shape[0]
    p_poisson = batch_priv_size / max(N, 1)
    mask = (torch.rand(N, device=device) < p_poisson)

    B_priv = X[mask]
    B_priv_pub = X_pub[mask]
    B_priv_labels = Y[mask]

    k = min(batch_pub_size, N)
    perm = torch.randperm(N, device=device)[:k]
    B_pub_pub = X_pub[perm]
    B_pub_labels = Y[perm]

    # Public gradient
    model.zero_grad()
    logits_pub, _ = model(B_pub_pub)
    loss_pub = loss_fn(logits_pub.squeeze(), B_pub_labels).mean()
    loss_pub.backward()
    grad_pub = [p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad()

    if B_priv.size(0) == 0:
        return grad_pub

    def _loss_fn(params, buffers, x, x_pub, y):
        logits_total, h_r = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        logits_pub_i, h_i = functional_call(model, (params, buffers), (x_pub.unsqueeze(0),))
        corr = rep_consistency_loss(h_r, h_i, C_h, beta).squeeze()
        task_loss = loss_fn(logits_total.squeeze(), y) - loss_fn(logits_pub_i.squeeze(), y)
        return task_loss + corr

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    grads = vmap(
        torch.func.grad(_loss_fn),
        in_dims=(None, None, 0, 0, 0),
        randomness="different",
    )(params, buffers, B_priv, B_priv_pub, B_priv_labels)

    B = B_priv.size(0)
    per_sample_norms_sq = torch.stack([
        g.view(B, -1).pow(2).sum(dim=1) for g in grads.values()
    ]).sum(dim=0)
    per_sample_norms = torch.sqrt(per_sample_norms_sq)
    clip_factors = (max_grad_norm / (per_sample_norms + 1e-6)).clamp(max=1.0)

    aggregated_grads = []
    for name, _ in model.named_parameters():
        g_batch = grads[name]
        view_shape = [B] + [1] * (g_batch.ndim - 1)
        factor = clip_factors.view(*view_shape)
        g_sum = (g_batch * factor).sum(dim=0)
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * max_grad_norm,
            size=g_sum.shape,
            device=g_sum.device,
        )
        aggregated_grads.append(g_sum / B + noise / B)

    total_grads = [grad_pub[i] + alpha * aggregated_grads[i] for i in range(len(aggregated_grads))]
    return total_grads

# ============================================================
# Feature construction
# ============================================================
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            # Bank dataset commonly uses "unknown" rather than "?"
            df.loc[df[c].isin(["?", "unknown"]), c] = np.nan
    return df

def _ensure_float32_contig(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))

def build_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].copy()

    # Bank label mapping: y is typically "yes"/"no"
    if y.dtype == object:
        s = y.astype(str).str.strip().str.lower()
        mapping = {"no": 0, "yes": 1}
        if s.isin(mapping.keys()).any():
            y = s.map(mapping)

    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    X = df.drop(columns=[target_col]).copy()
    return X, y

def build_noise_view(X_like: pd.DataFrame, sensitive_cols: List[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xn = X_like.copy()
    for c in sensitive_cols:
        if c in Xn.columns:
            if c in BANK_NUMERIC:
                Xn[c] = rng.standard_normal(size=len(Xn))
            elif c in BANK_CATEGORICAL:
                # Shuffle or replace with a constant/random category
                Xn[c] = rng.choice(Xn[c].dropna().unique(), size=len(Xn))
    return Xn

# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--support_csv", type=str, default="bank_support.csv")
    p.add_argument("--train_csv", type=str, default="bank_train.csv")
    p.add_argument("--train_imputed_csv", type=str, default="bank_train_imputed.csv")
    p.add_argument("--val_csv", type=str, default="bank_val.csv")
    p.add_argument("--test_csv", type=str, default="bank_test.csv")
    p.add_argument("--target_col", type=str, default="y")
    p.add_argument(
        "--sensitive_cols",
        type=str,
        nargs="+",
        default=["age", "job", "marital", "education", "housing"],
    )
    p.add_argument("--one_hot", action="store_true")
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--mode", type=str, default="dpsgd",
                   choices=["sgd_ori", "sgd_hybrid", "dpsgd", "feature-dp", "fusiondp"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--results_json", type=str, default="results_bank_seed.json")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"[ARGS] alpha={args.alpha} beta={args.beta} mode={args.mode} eps={args.epsilon} max_grad_norm={args.max_grad_norm}")
    set_seed(args.seed)

    # ---- Load split files ----
    df_support = _clean_df(pd.read_csv(args.support_csv))
    df_train   = _clean_df(pd.read_csv(args.train_csv))
    df_train_i = _clean_df(pd.read_csv(args.train_imputed_csv))
    df_val     = _clean_df(pd.read_csv(args.val_csv))
    df_test    = _clean_df(pd.read_csv(args.test_csv))

    target_col = args.target_col
    print(f"Target column: {target_col}")

    # Ensure columns match between train and train_imputed
    if set(df_train.columns) != set(df_train_i.columns):
        missing_in_imp = sorted(list(set(df_train.columns) - set(df_train_i.columns)))
        extra_in_imp   = sorted(list(set(df_train_i.columns) - set(df_train.columns)))
        raise ValueError(
            "train_csv and train_imputed_csv must have the same columns. "
            f"Missing in train_imputed: {missing_in_imp}. Extra in train_imputed: {extra_in_imp}."
        )
    df_train_i = df_train_i[df_train.columns.tolist()]  # same order

    # Build X/y for each split (real)
    Xtr_real, ytr = build_X_y(df_train, target_col)
    Xtr_imp, _    = build_X_y(df_train_i, target_col)   # imputed view; labels come from real train
    Xva_real, yva = build_X_y(df_val, target_col)
    Xte_real, yte = build_X_y(df_test, target_col)

    print("Xtr_real.shape:", Xtr_real.shape)

    # Noise view only needed for feature-dp
    Xtr_noise = build_noise_view(Xtr_imp, args.sensitive_cols, seed=args.seed)

    # One-hot: fit on TRAIN ONLY; apply same columns to val/test/imp/noise
    
    if args.one_hot:
        def split_num_cat(Xdf: pd.DataFrame):
            missing_num = [c for c in BANK_NUMERIC if c not in Xdf.columns]
            missing_cat = [c for c in BANK_CATEGORICAL if c not in Xdf.columns]
            if missing_num or missing_cat:
                raise ValueError(
                    "Bank schema columns missing.\n"
                    f"Missing numeric={missing_num}\n"
                    f"Missing categorical={missing_cat}\n"
                    "Check your CSV columns (bank_support/train/val/test)."
                )

            X_num = Xdf[BANK_NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X_cat = Xdf[BANK_CATEGORICAL].astype(str).fillna("__NA__")
            return X_num, X_cat

        


        Xtr_num, Xtr_cat = split_num_cat(Xtr_real)
        Xva_num, Xva_cat = split_num_cat(Xva_real)
        Xte_num, Xte_cat = split_num_cat(Xte_real)

        Xtr_i_num, Xtr_i_cat = split_num_cat(Xtr_imp)
        Xtr_n_num, Xtr_n_cat = split_num_cat(Xtr_noise)

        # Convert string "nan" to actual NaN in categorical columns
        def fix_string_nan(df_cat):
            for c in df_cat.columns:
                df_cat.loc[df_cat[c].str.lower() == 'nan', c] = np.nan
            return df_cat

        Xtr_cat = fix_string_nan(Xtr_cat)
        Xva_cat = fix_string_nan(Xva_cat)
        Xte_cat = fix_string_nan(Xte_cat)
        Xtr_i_cat = fix_string_nan(Xtr_i_cat)
        Xtr_n_cat = fix_string_nan(Xtr_n_cat)


        Xtr_cat_oh = pd.get_dummies(Xtr_cat, prefix=BANK_CATEGORICAL, dummy_na=True)
        cat_cols = Xtr_cat_oh.columns
        # print(cat_cols)

        Xva_cat_oh = pd.get_dummies(Xva_cat, prefix=BANK_CATEGORICAL, dummy_na=True).reindex(columns=cat_cols, fill_value=0)
        Xte_cat_oh = pd.get_dummies(Xte_cat, prefix=BANK_CATEGORICAL, dummy_na=True).reindex(columns=cat_cols, fill_value=0)
        Xtr_i_cat_oh = pd.get_dummies(Xtr_i_cat, prefix=BANK_CATEGORICAL, dummy_na=True).reindex(columns=cat_cols, fill_value=0)
        Xtr_n_cat_oh = pd.get_dummies(Xtr_n_cat, prefix=BANK_CATEGORICAL, dummy_na=True).reindex(columns=cat_cols, fill_value=0)
        
        Xtr_real  = pd.concat([Xtr_num, Xtr_cat_oh], axis=1)
        Xva_real  = pd.concat([Xva_num, Xva_cat_oh], axis=1)
        Xte_real  = pd.concat([Xte_num, Xte_cat_oh], axis=1)
        Xtr_imp   = pd.concat([Xtr_i_num, Xtr_i_cat_oh], axis=1)
        Xtr_noise = pd.concat([Xtr_n_num, Xtr_n_cat_oh], axis=1)

        print("After one-hot (num+cat):")
        print("  train:", Xtr_real.shape, " val:", Xva_real.shape, " test:", Xte_real.shape)
        print("  d =", Xtr_real.shape[1])

    # Standardize: fit on TRAIN REAL, then transform others
    scaler = StandardScaler()
    Xtr_real_np = scaler.fit_transform(Xtr_real.to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    Xva_real_np = scaler.transform(Xva_real.to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)
    Xte_real_np = scaler.transform(Xte_real.to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)

    Xtr_imp_np = Xtr_noise_np = None
    if args.mode in ["sgd_hybrid", "feature-dp", "fusiondp"]:
        Xtr_imp_np = scaler.transform(Xtr_imp.to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)

    if args.mode == "feature-dp":
        Xtr_noise_np = scaler.transform(Xtr_noise.to_numpy(dtype=np.float32, copy=False)).astype(np.float32, copy=False)

    Xtr_real_np = _ensure_float32_contig(Xtr_real_np)
    Xva_real_np = _ensure_float32_contig(Xva_real_np)
    Xte_real_np = _ensure_float32_contig(Xte_real_np)
    if Xtr_imp_np is not None:
        Xtr_imp_np = _ensure_float32_contig(Xtr_imp_np)
    if Xtr_noise_np is not None:
        Xtr_noise_np = _ensure_float32_contig(Xtr_noise_np)

    # DataLoaders
    train_loader = DataLoader(TabDataset(Xtr_real_np, ytr.values), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TabDataset(Xva_real_np, yva.values), batch_size=512, shuffle=False)
    test_loader  = DataLoader(TabDataset(Xte_real_np, yte.values), batch_size=512, shuffle=False)

    if args.mode == "sgd_hybrid":
        hybrid_loader = DataLoader(TabDataset(Xtr_imp_np, ytr.values), batch_size=args.batch_size, shuffle=True, drop_last=False)
    else:
        hybrid_loader = None

    # Model
    d_in = Xtr_real_np.shape[1]
    model = MLP(d_in).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Global tensors on GPU only for modes that use them
    X_tensor_g = X_pub_tensor_g = X_pub_tensor0_g = Y_tensor_g = None
    if args.mode in ["feature-dp", "fusiondp"]:
        if Xtr_imp_np is None:
            raise ValueError("Mode requires train_imputed features but Xtr_imp_np is None.")
        X_tensor_g     = torch.from_numpy(Xtr_real_np).to(DEVICE, non_blocking=True)
        X_pub_tensor_g = torch.from_numpy(Xtr_imp_np).to(DEVICE, non_blocking=True)
        Y_tensor_g     = torch.tensor(ytr.values, dtype=torch.float32, device=DEVICE)

        if args.mode == "feature-dp":
            if Xtr_noise_np is None:
                raise ValueError("feature-dp requires Xtr_noise_np but it is None")
            X_pub_tensor0_g = torch.from_numpy(Xtr_noise_np).to(DEVICE, non_blocking=True)

    # Noise multiplier
    noise_multiplier = 0.0
    if args.mode not in ["sgd_ori", "sgd_hybrid"]:
        sample_rate = args.batch_size / max(len(Xtr_real_np), 1)
        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            sample_rate=sample_rate,
            accountant="rdp",
            epochs=args.epochs,
        )
        print(f"Calculated Sigma: {noise_multiplier:.4f}")

    # Initialize C_h for fusiondp
    C_h = 5.0
    if args.mode == "fusiondp":
        with torch.no_grad():
            k = min(512, X_tensor_g.size(0))
            idx = torch.randperm(X_tensor_g.size(0), device=DEVICE)[:k]
            h_ori = model.get_rep(X_tensor_g[idx])
            h_imp = model.get_rep(X_pub_tensor_g[idx])
            C_h = (h_ori - h_imp).norm(dim=1).quantile(0.90).item()
        print(f"Initialized C_h: {C_h:.4f}")

    cur_result = {
        "mode": args.mode,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "alpha": args.alpha,
        "beta": args.beta,
        "one_hot": bool(args.one_hot),
        "target_col": target_col,
        "support_csv": args.support_csv,
        "train_csv": args.train_csv,
        "train_imputed_csv": args.train_imputed_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "sensitive_cols": args.sensitive_cols,
    }

    print(f"\nStarting Training Mode: {args.mode}")
    for epoch in range(args.epochs):
        model.train()

        if args.mode == "fusiondp":
            for _batch, _labels in train_loader:
                dp_grads = calibrate4(
                    model=model,
                    X=X_tensor_g,
                    X_pub=X_pub_tensor_g,
                    Y=Y_tensor_g,
                    loss_fn=loss_fn,
                    max_grad_norm=args.max_grad_norm,
                    noise_multiplier=noise_multiplier,
                    C_h=C_h,
                    beta=args.beta,
                    alpha=args.alpha,
                    batch_priv_size=args.batch_size,
                    batch_pub_size=4096,
                    device=str(DEVICE),
                )
                for i, p in enumerate(model.parameters()):
                    p.grad = dp_grads[i]
                optimizer.step()
                model.zero_grad()
                project_to_W([p.data for p in model.parameters()], M=10.0)

        elif args.mode == "feature-dp":
            for _ in train_loader:
                dp_grads = feature_dp_update_sample(
                    model=model,
                    X=X_tensor_g,
                    X_pub=X_pub_tensor0_g,
                    Y=Y_tensor_g,
                    loss_fn=loss_fn,
                    max_grad_norm=args.max_grad_norm,
                    noise_multiplier=noise_multiplier,
                    alpha=args.alpha,
                    M=10.0,
                    batch_priv_size=args.batch_size,
                    batch_pub_size=4096,
                )
                for i, p in enumerate(model.parameters()):
                    p.grad = dp_grads[i]
                optimizer.step()
                model.zero_grad()
                project_to_W([p.data for p in model.parameters()], M=10.0)

        elif args.mode == "dpsgd":
            for batch, labels in train_loader:
                batch = batch.to(DEVICE)
                labels = labels.to(DEVICE)

                dp_grads = dp_sgd_update(model, batch, labels, loss_fn, args.max_grad_norm, noise_multiplier)
                for i, p in enumerate(model.parameters()):
                    p.grad = dp_grads[i]
                optimizer.step()
                model.zero_grad()
                project_to_W([p.data for p in model.parameters()], M=10.0)

        elif args.mode == "sgd_ori":
            for batch, labels in train_loader:
                batch = batch.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits, _ = model(batch)
                loss = loss_fn(logits, labels).mean()
                loss.backward()
                optimizer.step()
                model.zero_grad()
                project_to_W([p.data for p in model.parameters()], M=10.0)

        elif args.mode == "sgd_hybrid":
            for batch, labels in hybrid_loader:
                batch = batch.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                logits, _ = model(batch)
                loss = loss_fn(logits, labels).mean()
                loss.backward()
                optimizer.step()
                model.zero_grad()
                project_to_W([p.data for p in model.parameters()], M=10.0)

        val_acc, val_auroc, val_aupr, val_f1 = evaluate(model, val_loader)
        te_acc, te_auroc, te_aupr, te_f1 = evaluate(model, test_loader)
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"VAL Acc:{val_acc:.4f} AUROC:{val_auroc:.4f} AUPR:{val_aupr:.4f} F1:{val_f1:.4f} || "
            f"TEST Acc:{te_acc:.4f} AUROC:{te_auroc:.4f} AUPR:{te_aupr:.4f} F1:{te_f1:.4f}"
        )

    # Save final (test) metrics
    cur_result["final_acc"] = te_acc
    cur_result["final_auroc"] = te_auroc
    cur_result["final_aupr"] = te_aupr
    cur_result["final_f1"] = te_f1
    cur_result["final_val_acc"] = val_acc
    cur_result["final_val_auroc"] = val_auroc
    cur_result["final_val_aupr"] = val_aupr
    cur_result["final_val_f1"] = val_f1

    with open(args.results_json, "a") as f:
        f.write(json.dumps(cur_result) + "\n")
    print("Results saved.")

if __name__ == "__main__":
    main()
