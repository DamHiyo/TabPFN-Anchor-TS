# -*- coding: utf-8 -*-
"""
run_tabpfn_m1.py

Paper3: TabPFN M1 (단일 모델) — unified rank pool top 500으로 평가
- Stacking 없이 M1 블록(500 features)만 TabPFN fit → predict
- 5-fold CV, 12 암종

실행 예시:
  python run_tabpfn_m1.py \
    --pool_method SHAP_RF \
    --fold_col R1:F1 \
    --device cuda:0
"""

import os
import sys
import re
import argparse
import time
import warnings
import gc
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, log_loss,
)

from config import Config
from data_loader import DataLoader

warnings.filterwarnings("ignore")


def _p(msg: str):
    print(msg, flush=True)


_ID_COLON_DIGITS = re.compile(r":(\d+):$")
_ID_UNDERSCORE = re.compile(r"_(\d+)_$")


def _extract_id(s):
    s = str(s)
    m = _ID_COLON_DIGITS.search(s)
    if m: return m.group(1)
    m = _ID_UNDERSCORE.search(s)
    return m.group(1) if m else None


def map_features(feats, x_cols):
    x_set = set(x_cols)
    id_to_col = {}
    for c in x_cols:
        cid = _extract_id(c)
        if cid and cid not in id_to_col:
            id_to_col[cid] = c
    mapped, seen = [], set()
    for rf in feats:
        rf = str(rf)
        if rf in x_set:
            col = rf
        else:
            rid = _extract_id(rf)
            if rid is None: continue
            col = id_to_col.get(rid)
            if col is None: continue
        if col not in seen:
            seen.add(col)
            mapped.append(col)
    return mapped


def make_tabpfn(device, n_estimators, seed):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    try:
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device=device,
            n_estimators=n_estimators, random_state=seed,
        )
    except TypeError:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        clf.set_params(device=device, n_estimators=n_estimators, random_state=seed)
        return clf


def _train_mask(cv_df, fold_col):
    s = pd.to_numeric(cv_df[fold_col], errors="coerce")
    return (s.to_numpy() == 0)


def run_one(dataset, fold_col, pool_csv, device, n_estimators, seed):
    _p(f"\n{'='*50}")
    _p(f"[TabPFN M1] {dataset} (fold={fold_col})")

    if not os.path.exists(pool_csv):
        _p(f"  Skipped: pool not found at {pool_csv}")
        return None

    df_pool = pd.read_csv(pool_csv)
    pool_feats_raw = df_pool["Feature"].astype(str).tolist()

    loader = DataLoader(dataset)
    X, y, cv_df = loader.load_data()
    if X is None: return None

    pool_feats = map_features(pool_feats_raw, X.columns.tolist())
    top500 = pool_feats[:500]
    _p(f"  Features: {len(top500)}")

    tr_mask = _train_mask(cv_df, fold_col)
    te_mask = ~tr_mask
    le = LabelEncoder()
    y_all = le.fit_transform(y.astype(str))
    n_classes = len(le.classes_)

    Xtr = X.loc[tr_mask, top500].values.astype(np.float32)
    Xte = X.loc[te_mask, top500].values.astype(np.float32)
    ytr = y_all[tr_mask]
    yte = y_all[te_mask]
    _p(f"  Train: {Xtr.shape}, Test: {Xte.shape}, Classes: {n_classes}")

    t0 = time.time()
    import torch
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    clf = make_tabpfn(device, n_estimators, seed)
    clf.fit(Xtr, ytr)
    train_time = time.time() - t0

    t0 = time.time()
    proba = clf.predict_proba(Xte).astype(np.float32)
    pred = np.argmax(proba, axis=1)
    infer_time = time.time() - t0

    # Align proba columns
    if proba.shape[1] < n_classes:
        aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float32)
        classes = clf.classes_ if hasattr(clf, 'classes_') else np.arange(proba.shape[1])
        for j, c in enumerate(classes):
            if 0 <= int(c) < n_classes:
                aligned[:, int(c)] = proba[:, j]
        proba = aligned

    met = {
        "dataset": dataset,
        "fold_col": fold_col,
        "n_features": len(top500),
        "n_train": Xtr.shape[0],
        "n_test": Xte.shape[0],
        "n_classes": n_classes,
        "accuracy": float(accuracy_score(yte, pred)),
        "precision_weighted": float(precision_score(yte, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(yte, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(yte, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }
    try:
        met["auroc_weighted_ovr"] = float(roc_auc_score(yte, proba, multi_class="ovr", average="weighted"))
    except: met["auroc_weighted_ovr"] = np.nan
    try:
        met["logloss"] = float(log_loss(yte, proba, labels=np.arange(n_classes)))
    except: met["logloss"] = np.nan

    _p(f"  F1={met['f1_weighted']:.4f} Acc={met['accuracy']:.4f} ({train_time:.1f}s)")

    del clf, Xtr, Xte
    gc.collect()
    return met


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=Config.ALL_DATASETS)
    parser.add_argument("--pool_method", type=str, required=True,
                        choices=["SHAP_RF", "SHAP_MI", "SHAP_EN"],
                        help="Feature selection method (SHAP_RF, SHAP_MI, SHAP_EN)")
    parser.add_argument("--fold_col", type=str, default=None,
                        help="Single fold (e.g., R1:F1). If omitted, runs all 5 folds.")
    parser.add_argument("--features_dir", type=str, default=None,
                        help="Features base dir. Default: results/features/{pool_method}/")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_estimators", type=int, default=32)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    args = parser.parse_args()

    folds = [args.fold_col] if args.fold_col else Config.FOLD_COLS

    if args.features_dir is None:
        args.features_dir = str(Config.RESULT_DIR / "features" / args.pool_method)
    if args.save_dir is None:
        args.save_dir = str(Config.RESULT_DIR / "tabpfn" / f"m1_{args.pool_method.lower()}")

    os.makedirs(args.save_dir, exist_ok=True)

    all_results = []
    for fold_col in folds:
        fold_tag = fold_col.replace(":", "")
        for i, ds in enumerate(args.datasets, 1):
            pool_csv = os.path.join(args.features_dir, fold_tag, ds, "unified_rank_pool.csv")
            _p(f"\n[{fold_col}] [{i}/{len(args.datasets)}] {ds}")
            try:
                met = run_one(ds, fold_col, pool_csv, args.device, args.n_estimators, args.seed)
                if met: all_results.append(met)
            except Exception as e:
                _p(f"  Failed: {e}")
                import traceback; traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        save_path = os.path.join(args.save_dir, "m1_results.csv")
        df.to_csv(save_path, index=False)
        _p(f"\n[Done] Saved: {save_path} ({len(df)} rows)")

        # Fold 평균 summary
        summary = df.groupby("dataset")[["accuracy", "f1_weighted", "balanced_accuracy", "auroc_weighted_ovr"]].agg(["mean", "std"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.to_csv(os.path.join(args.save_dir, "m1_summary.csv"), float_format="%.4f")
        _p(f"[Done] Summary: {os.path.join(args.save_dir, 'm1_summary.csv')}")


if __name__ == "__main__":
    try: sys.stdout.reconfigure(line_buffering=True)
    except: pass
    main()
