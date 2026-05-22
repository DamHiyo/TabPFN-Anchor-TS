# -*- coding: utf-8 -*-
"""
run_tabnet.py

TabNet baseline: 전체 feature (~20K) 사용, 5-fold CV
n_d/n_a = {16, 32, 64} 시도 후 best 결과 저장

실행:
  python run_tabnet.py --fold_col R1:F1 --device cuda:0
"""

import os
import sys
import argparse
import time
import warnings
import gc

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, log_loss,
)
from pytorch_tabnet.tab_model import TabNetClassifier

from config import Config
from data_loader import DataLoader

warnings.filterwarnings("ignore")


def _p(msg):
    print(msg, flush=True)


def _train_mask(cv_df, fold_col):
    s = pd.to_numeric(cv_df[fold_col], errors="coerce")
    return (s.to_numpy() == 0)


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    met = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            met["auroc_weighted_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except: met["auroc_weighted_ovr"] = np.nan
        try:
            met["logloss"] = float(log_loss(y_true, y_proba, labels=np.arange(n_classes)))
        except: met["logloss"] = np.nan
    return met


def run_one(dataset, fold_col, save_dir, device, seed):
    _p(f"\n{'='*50}")
    _p(f"[TabNet] {dataset} (fold={fold_col})")

    loader = DataLoader(dataset)
    X, y, cv_df = loader.load_data()
    if X is None: return []

    tr_mask = _train_mask(cv_df, fold_col)
    te_mask = ~tr_mask
    le = LabelEncoder()
    y_all = le.fit_transform(y.astype(str))
    n_classes = len(le.classes_)

    X_train = X.loc[tr_mask].values.astype(np.float64)
    X_test = X.loc[te_mask].values.astype(np.float64)
    y_train = y_all[tr_mask]
    y_test = y_all[te_mask]

    _p(f"  Train: {X_train.shape}, Test: {X_test.shape}, Classes: {n_classes}")

    results = []
    n_dims = [16, 32, 64]

    for nd in n_dims:
        _p(f"  [n_d={nd}] Training...")

        clf = TabNetClassifier(
            n_d=nd, n_a=nd,
            n_steps=5,
            gamma=1.5,
            lambda_sparse=1e-3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.02),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            device_name=device,
            verbose=0,
            seed=seed,
        )

        t0 = time.time()
        clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=['accuracy'],
            max_epochs=200,
            patience=20,
            batch_size=min(256, len(X_train)),
            drop_last=False,
        )
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test).astype(np.float32)
        infer_time = time.time() - t0

        # Align proba
        if y_proba.shape[1] < n_classes:
            aligned = np.zeros((y_proba.shape[0], n_classes), dtype=np.float32)
            for j in range(y_proba.shape[1]):
                aligned[:, j] = y_proba[:, j]
            y_proba = aligned

        met = compute_metrics(y_test, y_pred, y_proba, n_classes)
        met["model"] = f"TabNet_nd{nd}"
        met["dataset"] = dataset
        met["fold_col"] = fold_col
        met["n_d"] = nd
        met["n_train"] = X_train.shape[0]
        met["n_test"] = X_test.shape[0]
        met["n_features"] = X_train.shape[1]
        met["n_classes"] = n_classes
        met["train_time_sec"] = train_time
        met["infer_time_sec"] = infer_time
        met["best_epoch"] = clf.best_epoch if hasattr(clf, 'best_epoch') else -1

        results.append(met)
        _p(f"  [n_d={nd}] f1={met['f1_weighted']:.4f} acc={met['accuracy']:.4f} "
           f"epochs={met['best_epoch']} ({train_time:.1f}s)")

        del clf
        torch.cuda.empty_cache() if device.startswith("cuda") else None
        gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=Config.ALL_DATASETS)
    parser.add_argument("--fold_col", type=str, default=None,
                        help="Single fold. If omitted, runs all 5.")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=Config.SEED)
    args = parser.parse_args()

    folds = [args.fold_col] if args.fold_col else Config.FOLD_COLS

    if args.save_dir is None:
        args.save_dir = str(Config.RESULT_DIR / "tabnet")
    os.makedirs(args.save_dir, exist_ok=True)

    all_results = []
    for fold_col in folds:
        fold_tag = fold_col.replace(":", "")
        fold_dir = os.path.join(args.save_dir, fold_tag)
        os.makedirs(fold_dir, exist_ok=True)

        _p(f"\n===== {fold_col} =====")
        for i, ds in enumerate(args.datasets, 1):
            _p(f"\n[{fold_col}] [{i}/{len(args.datasets)}] {ds}")
            try:
                res = run_one(ds, fold_col, fold_dir, args.device, args.seed)
                if res:
                    all_results.extend(res)
                    # Per-dataset csv
                    pd.DataFrame(res).to_csv(
                        os.path.join(fold_dir, f"{ds}_tabnet.csv"), index=False
                    )
            except Exception as e:
                _p(f"  Failed: {e}")
                import traceback; traceback.print_exc()

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(args.save_dir, "tabnet_all.csv"), index=False)
        _p(f"\n[Done] Saved: {args.save_dir}/tabnet_all.csv ({len(df)} rows)")

        # Best n_d per dataset summary
        best = df.loc[df.groupby(['dataset', 'fold_col'])['f1_weighted'].idxmax()]
        best.to_csv(os.path.join(args.save_dir, "tabnet_best.csv"), index=False)
        _p(f"[Done] Best: {args.save_dir}/tabnet_best.csv")


if __name__ == "__main__":
    try: sys.stdout.reconfigure(line_buffering=True)
    except: pass
    main()
