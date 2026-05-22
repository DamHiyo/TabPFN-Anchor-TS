# -*- coding: utf-8 -*-
"""
step2_rf_feature_importance.py

Step 1-2: Refinement via Random Forest (2000 → 500)
- LR ranked top 2000 features에 대해 RF Gini importance 계산
- 출력: {save_dir}/{fold_name}/{dataset}_rf_ranked.csv

실행 예시:
  python step2_rf_feature_importance.py \
    --datasets BRCA COADREAD GEA KIRCKICH SKCM THCA UCEC BLCA HNSC LGGGBM LUAD LUSC \
    --fold_col R1:F1 \
    --lr_rank_dir ./artifacts/features/LR_OOF_NEGLOGLOSS_R1F1 \
    --save_dir ./artifacts/features/RF_FI_R1F1
"""

import os
import sys
import re
import argparse
import json
import time
import warnings
import gc
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import Config
from data_loader import DataLoader

warnings.filterwarnings("ignore")


def _p(msg: str):
    print(msg, flush=True)


# =========================================================
# Feature ID mapping (rank csv ↔ X columns)
# =========================================================
_ID_COLON_DIGITS = re.compile(r":(\d+):$")
_ID_UNDERSCORE = re.compile(r"_(\d+)_$")


def _extract_id(s: str) -> Optional[str]:
    s = str(s)
    m = _ID_COLON_DIGITS.search(s)
    if m:
        return m.group(1)
    m = _ID_UNDERSCORE.search(s)
    return m.group(1) if m else None


def map_rank_features_to_xcols(rank_feats: List[str], x_cols: List[str]) -> List[str]:
    x_set = set(x_cols)
    id_to_col: Dict[str, str] = {}
    for c in x_cols:
        cid = _extract_id(c)
        if cid and cid not in id_to_col:
            id_to_col[cid] = c

    mapped = []
    seen = set()
    for rf in rank_feats:
        rf = str(rf)
        if rf in x_set:
            col = rf
        else:
            rid = _extract_id(rf)
            if rid is None:
                continue
            col = id_to_col.get(rid)
            if col is None:
                continue
        if col in seen:
            continue
        seen.add(col)
        mapped.append(col)
    return mapped


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
    lr_rank_dir: str,
    save_dir: str,
    lr_top_n: int,
    rf_n_estimators: int,
    seed: int,
):
    _p(f"\n{'='*50}")
    _p(f"[Step 1-2] RF Feature Importance: {dataset} (fold={fold_col})")

    # Load LR ranking
    lr_csv = os.path.join(lr_rank_dir, f"{dataset}_ranked.csv")
    if not os.path.exists(lr_csv):
        _p(f"  Skipped {dataset}: LR rank not found at {lr_csv}")
        return
    df_lr = pd.read_csv(lr_csv)
    lr_feats_raw = df_lr["Feature"].astype(str).tolist()

    # Load data
    loader = DataLoader(dataset)
    X, y, cv_df = loader.load_data()
    if X is None:
        _p(f"  Skipped {dataset} (load failed)")
        return

    # Map features
    lr_feats = map_rank_features_to_xcols(lr_feats_raw, X.columns.tolist())
    top_feats = lr_feats[:lr_top_n]
    _p(f"  LR features mapped: {len(lr_feats)}, using top {len(top_feats)}")

    # Train split
    tr_mask = _train_mask(cv_df, fold_col)
    X_tr = X.loc[tr_mask, top_feats].values.astype(np.float32)

    le = LabelEncoder()
    y_tr = le.fit_transform(y.loc[tr_mask].astype(str))

    _p(f"  Train shape: {X_tr.shape}, classes: {len(le.classes_)}")

    # Fit RF
    t0 = time.perf_counter()
    rf = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    _p(f"  RF fit done ({elapsed:.1f}s)")

    # Gini importance
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]

    out_df = pd.DataFrame({
        "Feature": np.array(top_feats)[idx],
        "RF_Importance": imp[idx],
        "RF_Rank": np.arange(1, len(idx) + 1),
        "LR_Rank": [lr_feats.index(top_feats[i]) + 1 if top_feats[i] in lr_feats else -1 for i in idx],
    })

    save_path = os.path.join(save_dir, f"{dataset}_rf_ranked.csv")
    out_df.to_csv(save_path, index=False, float_format="%.18f")
    _p(f"  Saved: {save_path}")

    del rf, X_tr, y_tr
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=Config.ALL_DATASETS)
    parser.add_argument("--fold_col", type=str, default="R1:F1")
    parser.add_argument("--lr_rank_dir", type=str, required=True,
                        help="Directory with LR ranked CSVs from step1")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--lr_top_n", type=int, default=Config.LR_TOP_N)
    parser.add_argument("--rf_n_estimators", type=int, default=Config.RF_N_ESTIMATORS)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    args = parser.parse_args()

    if args.data_dir:
        Config.DATA_SOURCE_DIR = Path(args.data_dir)

    fold_tag = args.fold_col.replace(":", "")
    if args.save_dir is None:
        args.save_dir = str(Config.ARTIFACT_DIR / "features" / f"RF_FI_{fold_tag}")

    os.makedirs(args.save_dir, exist_ok=True)

    params = {
        "fold_col": args.fold_col,
        "datasets": args.datasets,
        "lr_rank_dir": args.lr_rank_dir,
        "lr_top_n": args.lr_top_n,
        "rf_n_estimators": args.rf_n_estimators,
        "seed": args.seed,
    }
    with open(os.path.join(args.save_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    for i, ds in enumerate(args.datasets, start=1):
        _p(f"\n[{i}/{len(args.datasets)}] {ds}")
        try:
            run_one_dataset(
                dataset=ds,
                fold_col=args.fold_col,
                lr_rank_dir=args.lr_rank_dir,
                save_dir=args.save_dir,
                lr_top_n=args.lr_top_n,
                rf_n_estimators=args.rf_n_estimators,
                seed=args.seed,
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
