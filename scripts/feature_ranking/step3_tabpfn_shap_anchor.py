# -*- coding: utf-8 -*-
"""
step3_tabpfn_shap_anchor.py

Step 1-3: Anchor Selection via TabPFN + SHAP (500 → 100)
+ Unified Rank Pool 생성 (SHAP > RF > LR priority)

- RF top 500 features에 대해 TabPFN(n_estimators=8) fit + SHAP
- SHAP 상위 100개 = Anchor Genes
- Unified Rank Pool: SHAP(F2) → RF(F1\F2) → LR(나머지)
- 출력:
  - {save_dir}/{dataset}/global_shap_rank.csv (SHAP ranking of F2)
  - {save_dir}/{dataset}/unified_rank_pool.csv (전체 통합 랭킹)
  - {save_dir}/{dataset}/anchor_genes.csv (top 100)

실행 예시:
  python step3_tabpfn_shap_anchor.py \
    --datasets BRCA COADREAD GEA KIRCKICH SKCM THCA UCEC BLCA HNSC LGGGBM LUAD LUSC \
    --fold_col R1:F1 \
    --lr_rank_dir ./artifacts/features/LR_OOF_NEGLOGLOSS_R1F1 \
    --rf_rank_dir ./artifacts/features/RF_FI_R1F1 \
    --save_dir ./artifacts/features/SHAP_ANCHOR_R1F1 \
    --device cuda:0
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
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from config import Config
from data_loader import DataLoader

warnings.filterwarnings("ignore")

# Thread limits for shared servers
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def _p(msg: str):
    print(msg, flush=True)


# =========================================================
# Feature ID mapping
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
    s = pd.to_numeric(cv_df[fold_col], errors="coerce")
    return (s.to_numpy() == 0)


# =========================================================
# TabPFN utilities
# =========================================================
def make_tabpfn(device: str, n_estimators: int, seed: int):
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    try:
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device=device,
            n_estimators=n_estimators, random_state=seed,
        )
    except TypeError:
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        try:
            clf.set_params(device=device, n_estimators=n_estimators, random_state=seed)
        except Exception:
            pass
        return clf


def safe_fit_tabpfn(Xtr, ytr, device, n_estimators, seed):
    import torch
    tries = [(device, n_estimators)]
    for ne in [4, 2]:
        if n_estimators > ne:
            tries.append((device, ne))
    if device.startswith("cuda"):
        tries.append(("cpu", min(n_estimators, 4)))

    for dev, ne in tries:
        try:
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            clf = make_tabpfn(dev, ne, seed)
            clf.fit(Xtr, ytr)
            return clf, (dev, ne)
        except Exception:
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
            continue
    raise RuntimeError("TabPFN fit failed after all retries")


def stratified_subsample(y_int, max_n, seed):
    n = len(y_int)
    if max_n <= 0 or max_n >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y_int, return_counts=True)
    frac = max_n / float(n)
    picked = []
    for c, cnt in zip(classes, counts):
        cand = np.where(y_int == c)[0]
        rng.shuffle(cand)
        picked.append(cand[:max(1, int(round(cnt * frac)))])
    picked = np.concatenate(picked)
    if len(picked) > max_n:
        rng.shuffle(picked)
        picked = picked[:max_n]
    return picked


def resolve_shap_getter():
    try:
        from tabpfn_extensions.interpretability.shap import get_shap_values
        return get_shap_values
    except Exception:
        pass
    raise ImportError("tabpfn_extensions SHAP backend not found.")


def reduce_shap_to_importance(shap_vals, n_samples, n_features):
    def _to_np(v):
        if hasattr(v, "values"):
            v = v.values
        return np.asarray(v)

    sv = shap_vals
    if isinstance(sv, (list, tuple)):
        arrs = [_to_np(x) for x in sv]
        if all(a.ndim == 2 for a in arrs):
            sv = np.stack(arrs, axis=2)
        else:
            sv = np.asarray(arrs)
    else:
        sv = _to_np(sv)

    if sv.ndim == 2:
        return np.mean(np.abs(sv), axis=0).astype(np.float64)
    elif sv.ndim == 3:
        if sv.shape[0] != n_samples and sv.shape[1] == n_samples:
            sv = np.transpose(sv, (1, 2, 0))
        return np.mean(np.abs(sv), axis=(0, 2)).astype(np.float64)
    raise ValueError(f"Unexpected SHAP shape: {sv.shape}")


# =========================================================
# Main logic
# =========================================================
def compute_shap_ranking(
    dataset: str,
    X_train: pd.DataFrame,
    y_train_int: np.ndarray,
    rf_top_feats: List[str],
    device: str,
    n_estimators: int,
    seed: int,
    shap_fit_max: int,
    shap_test_max: int,
    save_dir: str,
):
    """RF top 500 features에 대해 TabPFN SHAP 계산, global ranking 반환"""
    ds_dir = os.path.join(save_dir, dataset)
    os.makedirs(ds_dir, exist_ok=True)
    global_csv = os.path.join(ds_dir, "global_shap_rank.csv")

    if os.path.exists(global_csv):
        _p(f"  [SHAP] cached: {global_csv}")
        return pd.read_csv(global_csv)

    get_shap_values = resolve_shap_getter()

    # Fit TabPFN on full F2
    fit_idx = stratified_subsample(y_train_int, shap_fit_max, seed + 17)
    test_idx = stratified_subsample(y_train_int, shap_test_max, seed + 19)

    X_fit = X_train[rf_top_feats].values[fit_idx].astype(np.float32)
    y_fit = y_train_int[fit_idx]
    X_test_s = X_train[rf_top_feats].values[test_idx].astype(np.float32)

    _p(f"  [SHAP] fitting TabPFN (n_est={n_estimators}) on {len(rf_top_feats)} features...")
    t0 = time.perf_counter()
    clf, used = safe_fit_tabpfn(X_fit, y_fit, device, n_estimators, seed)
    _p(f"  [SHAP] fit done ({time.perf_counter()-t0:.1f}s), device={used[0]}, n_est={used[1]}")

    # SHAP
    _p(f"  [SHAP] computing SHAP values...")
    t0 = time.perf_counter()
    need_max_evals = 2 * len(rf_top_feats) + 1
    try:
        shap_vals = get_shap_values(
            clf, X_test_s,
            attribute_names=list(rf_top_feats),
            max_evals=need_max_evals,
        )
    except TypeError:
        try:
            shap_vals = get_shap_values(clf, X_test_s, max_evals=need_max_evals)
        except TypeError:
            shap_vals = get_shap_values(clf, X_test_s)

    imp = reduce_shap_to_importance(shap_vals, X_test_s.shape[0], len(rf_top_feats))
    _p(f"  [SHAP] done ({time.perf_counter()-t0:.1f}s)")

    del shap_vals, clf
    gc.collect()

    # Save SHAP ranking
    idx = np.argsort(imp)[::-1]
    df_shap = pd.DataFrame({
        "Feature": np.array(rf_top_feats)[idx],
        "ShapImportance": imp[idx],
        "Rank": np.arange(1, len(idx) + 1),
    })
    df_shap.to_csv(global_csv, index=False)
    _p(f"  [SHAP] saved: {global_csv}")

    # Save meta
    with open(os.path.join(ds_dir, "shap_meta.json"), "w") as f:
        json.dump({
            "dataset": dataset,
            "n_features": len(rf_top_feats),
            "n_estimators": n_estimators,
            "device_used": used[0],
            "n_estimators_used": used[1],
            "fit_samples": len(fit_idx),
            "test_samples": len(test_idx),
        }, f, indent=2)

    return df_shap


def build_unified_rank_pool(
    dataset: str,
    shap_df: pd.DataFrame,
    rf_feats: List[str],
    lr_feats: List[str],
    anchor_n: int,
    save_dir: str,
):
    """
    Unified Rank Pool 생성:
    1. SHAP ranked F2 (500 features) - SHAP importance 순
    2. RF ranked (F1 \ F2) - RF importance 순
    3. LR ranked (나머지) - LR importance 순
    """
    ds_dir = os.path.join(save_dir, dataset)
    pool_csv = os.path.join(ds_dir, "unified_rank_pool.csv")
    anchor_csv = os.path.join(ds_dir, "anchor_genes.csv")

    # Section 1: SHAP ranked F2
    shap_feats = shap_df["Feature"].astype(str).tolist()
    shap_set = set(shap_feats)

    # Section 2: RF ranked features NOT in F2
    rf_remaining = [f for f in rf_feats if f not in shap_set]
    section2_set = set(rf_remaining)

    # Section 3: LR ranked features NOT in F1 (not in shap + not in rf_remaining)
    used = shap_set | section2_set
    lr_remaining = [f for f in lr_feats if f not in used]

    # Build pool
    pool = []
    rank = 1
    for f in shap_feats:
        pool.append({"Feature": f, "Pool_Rank": rank, "Source": "SHAP_F2"})
        rank += 1
    for f in rf_remaining:
        pool.append({"Feature": f, "Pool_Rank": rank, "Source": "RF_F1_minus_F2"})
        rank += 1
    for f in lr_remaining:
        pool.append({"Feature": f, "Pool_Rank": rank, "Source": "LR_rest"})
        rank += 1

    df_pool = pd.DataFrame(pool)
    df_pool.to_csv(pool_csv, index=False)

    # Anchor genes = SHAP top 100
    anchor_feats = shap_feats[:anchor_n]
    df_anchor = pd.DataFrame({
        "Feature": anchor_feats,
        "Anchor_Rank": range(1, len(anchor_feats) + 1),
    })
    df_anchor.to_csv(anchor_csv, index=False)

    _p(f"  [Pool] SHAP_F2={len(shap_feats)}, RF(F1\\F2)={len(rf_remaining)}, LR(rest)={len(lr_remaining)}")
    _p(f"  [Pool] Total pool: {len(pool)}, Anchors: {len(anchor_feats)}")
    _p(f"  Saved: {pool_csv}")
    _p(f"  Saved: {anchor_csv}")

    return df_pool, anchor_feats


def run_one_dataset(
    dataset: str,
    fold_col: str,
    lr_rank_dir: str,
    rf_rank_dir: str,
    save_dir: str,
    device: str,
    args,
):
    _p(f"\n{'='*50}")
    _p(f"[Step 1-3] TabPFN SHAP Anchor: {dataset} (fold={fold_col})")

    # Load LR ranking (full)
    lr_csv = os.path.join(lr_rank_dir, f"{dataset}_ranked.csv")
    df_lr = pd.read_csv(lr_csv)
    lr_feats_raw = df_lr["Feature"].astype(str).tolist()

    # Load RF ranking
    rf_csv = os.path.join(rf_rank_dir, f"{dataset}_rf_ranked.csv")
    df_rf = pd.read_csv(rf_csv)
    rf_feats_raw = df_rf["Feature"].astype(str).tolist()

    # Load data
    loader = DataLoader(dataset)
    X, y, cv_df = loader.load_data()
    if X is None:
        _p(f"  Skipped {dataset}")
        return

    # Map features to X columns
    lr_feats = map_rank_features_to_xcols(lr_feats_raw, X.columns.tolist())
    rf_feats = map_rank_features_to_xcols(rf_feats_raw, X.columns.tolist())
    rf_top_feats = rf_feats[:Config.RF_TOP_N]  # top 500

    _p(f"  LR mapped: {len(lr_feats)}, RF mapped: {len(rf_feats)}, RF top {Config.RF_TOP_N}: {len(rf_top_feats)}")

    # Train split
    tr_mask = _train_mask(cv_df, fold_col)
    X_train = X.loc[tr_mask]
    le = LabelEncoder()
    y_train_int = le.fit_transform(y.loc[tr_mask].astype(str)).astype(np.int64)

    # SHAP ranking on RF top 500
    shap_df = compute_shap_ranking(
        dataset=dataset,
        X_train=X_train,
        y_train_int=y_train_int,
        rf_top_feats=rf_top_feats,
        device=device,
        n_estimators=Config.SHAP_N_ESTIMATORS,
        seed=Config.SEED,
        shap_fit_max=Config.SHAP_FIT_MAX_SAMPLES,
        shap_test_max=Config.SHAP_TEST_MAX_SAMPLES,
        save_dir=save_dir,
    )

    # Build unified rank pool
    build_unified_rank_pool(
        dataset=dataset,
        shap_df=shap_df,
        rf_feats=rf_feats,
        lr_feats=lr_feats,
        anchor_n=Config.ANCHOR_N,
        save_dir=save_dir,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=Config.ALL_DATASETS)
    parser.add_argument("--fold_col", type=str, default="R1:F1")
    parser.add_argument("--lr_rank_dir", type=str, required=True)
    parser.add_argument("--rf_rank_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.data_dir:
        Config.DATA_SOURCE_DIR = Path(args.data_dir)

    fold_tag = args.fold_col.replace(":", "")
    if args.save_dir is None:
        args.save_dir = str(Config.ARTIFACT_DIR / "features" / f"SHAP_ANCHOR_{fold_tag}")

    os.makedirs(args.save_dir, exist_ok=True)

    for i, ds in enumerate(args.datasets, start=1):
        _p(f"\n[{i}/{len(args.datasets)}] {ds}")
        try:
            run_one_dataset(
                dataset=ds,
                fold_col=args.fold_col,
                lr_rank_dir=args.lr_rank_dir,
                rf_rank_dir=args.rf_rank_dir,
                save_dir=args.save_dir,
                device=args.device,
                args=args,
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
