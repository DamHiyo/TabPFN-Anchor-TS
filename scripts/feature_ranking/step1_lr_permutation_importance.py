# -*- coding: utf-8 -*-
"""
step1_lr_permutation_importance.py

Step 1-1: Coarse Filtering via Logistic Regression (P → 2000)
- 각 fold의 train split에서 LR + Permutation Importance (neg_log_loss) 계산
- OOF 방식: train 내부 StratifiedKFold로 PI 집계
- 출력: {save_dir}/{fold_name}/{dataset}_ranked.csv

실행 예시:
  python step1_lr_permutation_importance.py \
    --datasets BRCA COADREAD GEA KIRCKICH SKCM THCA UCEC BLCA HNSC LGGGBM LUAD LUSC \
    --fold_col R1:F1 \
    --save_dir ./artifacts/features/LR_OOF_NEGLOGLOSS_R1_F1
"""

import os
import sys
import argparse
import json
import time
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

from config import Config
from data_loader import DataLoader

warnings.filterwarnings("ignore")


def _p(msg: str):
    print(msg, flush=True)


LR_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "tol": 1e-6,
    "max_iter": 10000,
    "random_state": Config.SEED,
    "n_jobs": 1,
}


def _encode_labels(y: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(y):
        return y.to_numpy()
    le = LabelEncoder()
    return le.fit_transform(y.astype(str))


def _train_mask(cv_df: pd.DataFrame, fold_col: str) -> np.ndarray:
    if fold_col not in cv_df.columns:
        raise RuntimeError(f"CV file does not contain column {fold_col}")
    s = pd.to_numeric(cv_df[fold_col], errors="coerce")
    mask = (s.to_numpy() == 0)
    if int(mask.sum()) == 0:
        raise RuntimeError(f"no train samples where {fold_col} == 0")
    return mask


def run_one_dataset(
    dataset: str,
    fold_col: str,
    save_dir: str,
    repeats: int,
    pi_n_jobs: int,
    n_splits: int,
    x_dtype: str,
):
    _p(f"\n{'='*50}")
    _p(f"[Step 1-1] LR Permutation Importance: {dataset} (fold={fold_col})")

    loader = DataLoader(dataset)
    X, y, cv_df = loader.load_data()
    if X is None:
        _p(f"  Skipped {dataset} (load failed)")
        return

    tr_mask = _train_mask(cv_df, fold_col)
    X_tr_df = X.loc[tr_mask]
    y_tr = y.loc[tr_mask]
    y_enc = _encode_labels(y_tr)
    feat_names = X_tr_df.columns.to_numpy()

    X_tr = np.ascontiguousarray(X_tr_df.to_numpy(dtype=np.dtype(x_dtype), copy=True))
    _p(f"  Train shape: {X_tr.shape}, dtype={X_tr.dtype}")

    # effective n_splits
    counts = np.bincount(y_enc.astype(int))
    counts = counts[counts > 0]
    eff = min(n_splits, int(counts.min())) if counts.size > 0 else 0
    if eff < 2:
        _p(f"  Skipped {dataset}: too few samples in smallest class")
        return
    if eff != n_splits:
        _p(f"  Warning: n_splits reduced from {n_splits} to {eff}")

    t0 = time.perf_counter()
    skf = StratifiedKFold(n_splits=eff, shuffle=True, random_state=Config.SEED)

    n_features = X_tr.shape[1]
    mean_sum = np.zeros(n_features, dtype=np.float64)
    std_sum = np.zeros(n_features, dtype=np.float64)
    weight_sum = 0.0

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_enc), start=1):
        _p(f"  [Fold {fold_idx}/{eff}] train={len(tr_idx)} val={len(va_idx)}")

        scaler = StandardScaler()
        X_fold_tr_sc = scaler.fit_transform(X_tr[tr_idx])
        X_fold_va_sc = scaler.transform(X_tr[va_idx])

        X_fold_tr_sc = np.ascontiguousarray(X_fold_tr_sc.astype(np.dtype(x_dtype), copy=False))
        X_fold_va_sc = np.ascontiguousarray(X_fold_va_sc.astype(np.dtype(x_dtype), copy=False))

        model = LogisticRegression(**LR_PARAMS)
        model.fit(X_fold_tr_sc, y_enc[tr_idx])

        result = permutation_importance(
            model, X_fold_va_sc, y_enc[va_idx],
            n_repeats=repeats,
            random_state=Config.SEED,
            n_jobs=pi_n_jobs,
            scoring="neg_log_loss",
        )

        w = float(len(va_idx))
        mean_sum += w * result.importances_mean
        std_sum += w * result.importances_std
        weight_sum += w

        del model, result, scaler
        gc.collect()

    imp_mean = mean_sum / max(weight_sum, 1.0)
    imp_std = std_sum / max(weight_sum, 1.0)

    idx = np.argsort(imp_mean)[::-1]
    save_path = os.path.join(save_dir, f"{dataset}_ranked.csv")
    out_df = pd.DataFrame({
        "Feature": feat_names[idx],
        "Importance_Mean": imp_mean[idx],
        "Importance_Std": imp_std[idx],
        "Rank": np.arange(1, len(idx) + 1),
    })
    out_df.to_csv(save_path, index=False, float_format="%.18f")

    elapsed = time.perf_counter() - t0
    _p(f"  Saved: {save_path} ({elapsed:.1f}s)")

    del X_tr, y_enc, feat_names, out_df
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=Config.ALL_DATASETS)
    parser.add_argument("--fold_col", type=str, default="R1:F1",
                        help="CV fold column (e.g., R1:F1, R1:F2, ...)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory. Default: artifacts/features/LR_OOF_NEGLOGLOSS_{fold_col}")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=Config.LR_PERM_REPEATS)
    parser.add_argument("--pi_n_jobs", type=int, default=64)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--x_dtype", type=str, default="float32")
    args = parser.parse_args()

    if args.data_dir:
        Config.DATA_SOURCE_DIR = Path(args.data_dir)

    fold_tag = args.fold_col.replace(":", "")
    if args.save_dir is None:
        args.save_dir = str(Config.ARTIFACT_DIR / "features" / f"LR_OOF_NEGLOGLOSS_{fold_tag}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Save params
    params = {
        "fold_col": args.fold_col,
        "datasets": args.datasets,
        "repeats": args.repeats,
        "n_splits": args.n_splits,
        "scoring": "neg_log_loss",
        "lr_params": LR_PARAMS,
        "seed": Config.SEED,
    }
    with open(os.path.join(args.save_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    for i, ds in enumerate(args.datasets, start=1):
        _p(f"\n[{i}/{len(args.datasets)}] {ds}")
        try:
            run_one_dataset(
                dataset=ds,
                fold_col=args.fold_col,
                save_dir=args.save_dir,
                repeats=args.repeats,
                pi_n_jobs=args.pi_n_jobs,
                n_splits=args.n_splits,
                x_dtype=args.x_dtype,
            )
        except Exception as e:
            _p(f"  Failed {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
