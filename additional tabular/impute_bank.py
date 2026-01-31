#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch

from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier, TabPFNRegressor

# ===========================
# 1) Configuration
# ===========================
SEED = 66

# TabPFN settings
SUPPORT_SIZE = 512
BATCH_SIZE = 1024
N_ESTIMATORS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Split sizes (bank-additional-full has 41188 rows)
SUPPORT_N = 512
TRAIN_N   = 30000
VAL_N     = 5000
TEST_N    = 5000

OTHER = "__OTHER__"

# Fixed top-k policies for FEATURES (fit on TRAIN real columns; applied to ALL splits)
# Bank Marketing has moderate cardinality; cap a couple of columns to stabilize RAM / one-hot width.
FEATURE_TOPK_POLICY = {
    "job": 9,
    # optional safety (usually not needed, but harmless):
    # "education": 9,
}

# ---------------------------
# Sensitive feature choices
# ---------------------------
# A reasonable privacy/fairness-ish split (demographic + socioeconomic)
SENSITIVE_TARGETS = [
    "age",        # numeric
    "job",        # categorical
    "marital",    # categorical
    "education",  # categorical
    "housing",    # categorical (has housing loan)
]

NUM_TARGETS = ["age"]
CAT_TARGETS = [c for c in SENSITIVE_TARGETS if c not in NUM_TARGETS]


# ===========================
# 2) Helpers (keep model logic same)
# ===========================
def one_hot_align(query_df: pd.DataFrame, support_df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    One-hot encode predictors for query and support, and align columns.
    Returns (X_support, X_query) as numpy arrays.
    """
    sup = pd.get_dummies(support_df[cols], dummy_na=True)
    qry = pd.get_dummies(query_df[cols], dummy_na=True)
    sup, qry = sup.align(qry, join="outer", axis=1, fill_value=0)
    return sup.values, qry.values


def _clean_object_col(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    # Bank Marketing uses "unknown" frequently; treat it as missing for imputation consistency.
    s = s.replace({"?": np.nan, "nan": np.nan, "None": np.nan, "unknown": np.nan})
    return s


def fit_topk_vocab(series: pd.Series, k: int, other_label: str = OTHER) -> set[str]:
    """
    Fit a fixed vocab from a reference series (typically TRAIN).
    Keeps top-k frequent categories + OTHER.
    """
    s = _clean_object_col(series).fillna(other_label)
    vc = s[s != other_label].value_counts()
    kept = set(vc.index[:k].tolist())
    kept.add(other_label)
    return kept


def apply_vocab(series: pd.Series, vocab: set[str], other_label: str = OTHER) -> pd.Series:
    """
    Map any category not in vocab to OTHER, consistently.
    """
    s = _clean_object_col(series).fillna(other_label)
    return s.where(s.isin(vocab), other_label)


def build_fixed_vocabs_from_train(df_train: pd.DataFrame) -> dict[str, set[str]]:
    """
    Build fixed vocabs for specified categorical columns from TRAIN only.
    """
    vocabs: dict[str, set[str]] = {}
    for col, k in FEATURE_TOPK_POLICY.items():
        if col not in df_train.columns:
            continue
        if df_train[col].dtype != object:
            continue
        vocabs[col] = fit_topk_vocab(df_train[col], k=k, other_label=OTHER)
    return vocabs


def apply_fixed_vocabs_to_splits(
    df_support: pd.DataFrame,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    vocabs: dict[str, set[str]],
) -> None:
    """
    Apply fixed vocabs (fit on TRAIN) to all splits, in-place.
    Applies to REAL columns only.
    """
    for df in (df_support, df_train, df_val, df_test):
        for col, vocab in vocabs.items():
            if col in df.columns:
                df[col] = apply_vocab(df[col], vocab, other_label=OTHER)


def impute_column(
    df_support: pd.DataFrame,
    df_query: pd.DataFrame,
    target_col: str,
    public_cols: list[str],
) -> np.ndarray:
    print(f"\n--- Imputing Sensitive Feature: {target_col} ---")

    # 1) Prepare predictors (one-hot for TabPFN)
    X_support, X_query = one_hot_align(df_query, df_support, public_cols)

    # 2) Targets (raw)
    y_support_raw = df_support[target_col].values
    y_true_query_raw = df_query[target_col].values

    # 3) Choose model
    if target_col in CAT_TARGETS:
        print("   Type: Categorical -> Using TabPFNClassifier")

        sup_s = _clean_object_col(df_support[target_col]).fillna(OTHER)
        qry_s = _clean_object_col(df_query[target_col]).fillna(OTHER)
        y_support_raw = sup_s.values
        y_true_query_raw = qry_s.values

        le = LabelEncoder()
        y_support = le.fit_transform(pd.Series(y_support_raw).astype(str))

        model = TabPFNClassifier(device=DEVICE, n_estimators=N_ESTIMATORS)
        is_regression = False

    else:
        print("   Type: Numerical -> Using TabPFNRegressor")
        # Age is numeric; coerce, fill NaN with median to avoid crashes
        y_support = pd.to_numeric(df_support[target_col], errors="coerce")
        y_support = y_support.fillna(y_support.median()).astype(float).values

        model = TabPFNRegressor(device=DEVICE, n_estimators=N_ESTIMATORS)
        is_regression = True

    # 4) Subsample support (TabPFN comfort zone)
    if len(X_support) > SUPPORT_SIZE:
        X_support = X_support[:SUPPORT_SIZE]
        y_support = y_support[:SUPPORT_SIZE]

    # 5) Fit
    model.fit(X_support, y_support)

    # 6) Predict in batches
    preds = []
    print(f"   Predicting {len(X_query)} rows in batches of {BATCH_SIZE}...")
    for i in range(0, len(X_query), BATCH_SIZE):
        batch_X = X_query[i : i + BATCH_SIZE]
        batch_preds = model.predict(batch_X)
        preds.append(batch_preds)
    y_pred = np.concatenate(preds)

    # 7) Convert back for categorical + optional eval (on TRAIN query only)
    if not is_regression:
        y_pred_decoded = le.inverse_transform(y_pred.astype(int))

        acc = accuracy_score(pd.Series(y_true_query_raw).astype(str), pd.Series(y_pred_decoded).astype(str))
        f1 = f1_score(pd.Series(y_true_query_raw).astype(str), pd.Series(y_pred_decoded).astype(str), average="macro")
        print(f"✅ {target_col} Imputation Stats (on TRAIN query):")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Macro F1: {f1:.4f}")
        return y_pred_decoded
    else:
        y_true_num = pd.to_numeric(df_query[target_col], errors="coerce")
        y_true_num = y_true_num.fillna(y_true_num.median()).astype(float).values

        mse = mean_squared_error(y_true_num, y_pred.astype(float))
        r2 = r2_score(y_true_num, y_pred.astype(float))
        print(f"✅ {target_col} Imputation Stats (on TRAIN query):")
        print(f"   MSE: {mse:.4f}")
        print(f"   R2:  {r2:.4f}")
        return y_pred.astype(float)


# ===========================
# 3) Main
# ===========================
if __name__ == "__main__":
    np.random.seed(SEED)

    print("Fetching Bank Marketing (UCI id=222)...")
    ds = fetch_ucirepo(id=222)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    # Clean object columns + map "unknown"/"?" to NaN
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = _clean_object_col(X[c])
            X.loc[X[c].isna(), c] = np.nan

    for c in y.columns:
        if y[c].dtype == "object":
            y[c] = _clean_object_col(y[c])
            y.loc[y[c].isna(), c] = np.nan

    X.columns = [str(c) for c in X.columns]
    y.columns = [str(c) for c in y.columns]

    print(f"Dataset loaded: X={X.shape}, y={y.shape}")
    print("Columns:", list(X.columns))
    print("Target columns:", list(y.columns))

    # Bank target is usually named "y"
    if "y" not in y.columns:
        raise ValueError(f"Expected target column 'y' in y, got {list(y.columns)}")

    # Validate sensitive cols exist
    missing = [c for c in SENSITIVE_TARGETS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing sensitive columns: {missing}")

    # Public predictors (exclude sensitive targets)
    PUBLIC_FEATURES = [c for c in X.columns if c not in SENSITIVE_TARGETS]

    # ---------------------------
    # Split: support/train/val/test
    # ---------------------------
    n = len(X)
    idx = np.random.permutation(n)

    if SUPPORT_N + TRAIN_N + VAL_N + TEST_N > n:
        raise ValueError(
            f"Split sizes too large for dataset size n={n}. "
            f"Requested={SUPPORT_N + TRAIN_N + VAL_N + TEST_N}."
        )

    sup_idx  = idx[:SUPPORT_N]
    tr_idx   = idx[SUPPORT_N : SUPPORT_N + TRAIN_N]
    val_idx  = idx[SUPPORT_N + TRAIN_N : SUPPORT_N + TRAIN_N + VAL_N]
    test_idx = idx[SUPPORT_N + TRAIN_N + VAL_N :
                   SUPPORT_N + TRAIN_N + VAL_N + TEST_N]

    df_support = X.iloc[sup_idx].reset_index(drop=True)
    df_train   = X.iloc[tr_idx].reset_index(drop=True)
    df_val     = X.iloc[val_idx].reset_index(drop=True)
    df_test    = X.iloc[test_idx].reset_index(drop=True)

    # Attach labels (support has no y, consistent with your Adult script)
    df_train["y"] = y.iloc[tr_idx].reset_index(drop=True)["y"]
    df_val["y"]   = y.iloc[val_idx].reset_index(drop=True)["y"]
    df_test["y"]  = y.iloc[test_idx].reset_index(drop=True)["y"]

    # ---------------------------
    # Cap categorical FEATURE cardinality per FEATURE_TOPK_POLICY
    # Fit vocab on TRAIN only, apply to all splits.
    # ---------------------------
    feature_vocabs = build_fixed_vocabs_from_train(df_train)
    apply_fixed_vocabs_to_splits(df_support, df_train, df_val, df_test, feature_vocabs)

    # ---------------------------
    # Create train_imputed:
    # - copy TRAIN
    # - impute ONLY the sensitive features using SUPPORT as reference
    # - replace the ORIGINAL sensitive columns (no new columns)
    # ---------------------------
    df_train_imputed = df_train.copy()

    for target in SENSITIVE_TARGETS:
        print(f"\n=== Imputing {target} into train_imputed ===")
        imputed_train = impute_column(df_support, df_train_imputed, target, PUBLIC_FEATURES)

        # Replace original column (no new features created)
        df_train_imputed[target] = imputed_train

        # If this column is capped, re-apply vocab so it stays within top-k+OTHER
        if target in feature_vocabs:
            df_train_imputed[target] = apply_vocab(df_train_imputed[target], feature_vocabs[target], other_label=OTHER)

    # ---------------------------
    # Save CSVs
    # ---------------------------
    df_support.to_csv("bank_support.csv", index=False)
    df_train.to_csv("bank_train.csv", index=False)
    df_train_imputed.to_csv("bank_train_imputed.csv", index=False)
    df_val.to_csv("bank_val.csv", index=False)
    df_test.to_csv("bank_test.csv", index=False)

    print("\n✅ Saved datasets:")
    print(" - bank_support.csv")
    print(" - bank_train.csv")
    print(" - bank_train_imputed.csv")
    print(" - bank_val.csv")
    print(" - bank_test.csv")
