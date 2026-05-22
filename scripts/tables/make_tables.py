# -*- coding: utf-8 -*-
"""
make_tables.py

모든 Paper Table 생성 (Table 3~14)
- 5-fold 결과 집계
- 각 Table별 CSV 생성

실행 예시:
  python make_tables.py \
    --result_dirs ./final_results/R1_F1/stacking_v2 \
                  ./final_results/R1_F2/stacking_v2 \
                  ./final_results/R1_F3/stacking_v2 \
                  ./final_results/R1_F4/stacking_v2 \
                  ./final_results/R1_F5/stacking_v2 \
    --ml_dirs ./final_results/ml_baselines/R1F1 \
              ./final_results/ml_baselines/R1F2 \
              ./final_results/ml_baselines/R1F3 \
              ./final_results/ml_baselines/R1F4 \
              ./final_results/ml_baselines/R1F5 \
    --anchor_dirs ./artifacts/features/SHAP_ANCHOR_R1F1 \
                  ./artifacts/features/SHAP_ANCHOR_R1F2 \
                  ./artifacts/features/SHAP_ANCHOR_R1F3 \
                  ./artifacts/features/SHAP_ANCHOR_R1F4 \
                  ./artifacts/features/SHAP_ANCHOR_R1F5 \
    --save_dir ./paper_tables
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data_loader import DataLoader

warnings.filterwarnings("ignore")


def _p(msg: str):
    print(msg, flush=True)


# =========================================================
# Table 3/4: Dataset Summary & Subtype Distribution
# =========================================================
def make_table3_4(save_dir):
    """Dataset summary from data files"""
    _p("[Table 3/4] Dataset Summary & Subtype Distribution")
    rows3 = []
    rows4 = []

    for ds in Config.ALL_DATASETS:
        loader = DataLoader(ds)
        X, y, cv_df = loader.load_data()
        if X is None:
            continue

        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_classes = y.nunique()

        rows3.append({
            "Dataset": ds,
            "Samples": n_samples,
            "Features": n_features,
            "Subtypes": n_classes,
        })

        # Subtype distribution
        vc = y.value_counts().sort_values(ascending=False)
        row4 = {"Dataset": ds}
        for i, (label, count) in enumerate(vc.items(), start=1):
            row4[f"Class_{i}"] = f"{label} ({count})"
        max_count = vc.iloc[0]
        min_count = vc.iloc[-1]
        row4["Imbalance_Ratio"] = f"{max_count/min_count:.1f}:1"
        rows4.append(row4)

    df3 = pd.DataFrame(rows3)
    df3.to_csv(os.path.join(save_dir, "table3_dataset_summary.csv"), index=False)
    _p(f"  Saved table3")

    df4 = pd.DataFrame(rows4)
    df4.to_csv(os.path.join(save_dir, "table4_subtype_distribution.csv"), index=False)
    _p(f"  Saved table4")


# =========================================================
# Table 7: Ensemble Convergence (K analysis)
# =========================================================
def make_table7(result_dirs, save_dir):
    """K convergence from metrics_by_k.csv across folds"""
    _p("[Table 7] Ensemble Convergence")

    all_dfs = []
    for rd in result_dirs:
        csv = os.path.join(rd, "metrics_by_k.csv")
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            all_dfs.append(df)

    if not all_dfs:
        _p("  No metrics_by_k.csv found")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    # Average across folds per (dataset, k)
    target_ks = [1, 5, 10, 15, 20, 30]
    df_filt = df_all[df_all["k"].isin(target_ks)]

    # Mean across folds
    agg = df_filt.groupby(["dataset", "k"]).agg({
        "f1_weighted": "mean",
        "train_f1_weighted": "mean",
        "accuracy": "mean",
        "auroc_weighted_ovr": "mean",
    }).reset_index()

    # Wide format
    pivot = agg.pivot_table(index="k", columns="dataset", values="f1_weighted")
    pivot["Average"] = pivot.mean(axis=1)
    pivot.to_csv(os.path.join(save_dir, "table7_convergence_f1.csv"))
    _p(f"  Saved table7")

    # Train vs Test
    train_pivot = agg.pivot_table(index="k", columns="dataset", values="train_f1_weighted")
    train_pivot["Average"] = train_pivot.mean(axis=1)
    train_pivot.to_csv(os.path.join(save_dir, "table7_convergence_train_f1.csv"))


# =========================================================
# Table 9/10: Training Time & Inference Latency
# =========================================================
def make_table9_10(result_dirs, save_dir, k_target=15):
    """Training time breakdown and inference latency"""
    _p("[Table 9/10] Training Time & Inference Latency")

    time_rows = []
    latency_rows = []

    for rd in result_dirs:
        csv = os.path.join(rd, "metrics_by_k.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)

        for ds in Config.ALL_DATASETS:
            sub = df[(df["dataset"] == ds) & (df["k"] == k_target)]
            if sub.empty:
                continue
            row = sub.iloc[0]

            # Time: sum of base block elapsed
            base_dir = Path(rd) / "base_npz" / "unified_pool" / ds
            total_base_sec = 0.0
            total_latency_ms = 0.0
            single_latency = np.nan
            for i in range(1, k_target + 1):
                npz = base_dir / f"M{i:03d}.npz"
                if npz.exists():
                    d = np.load(npz, allow_pickle=True)
                    total_base_sec += float(d.get("elapsed_sec", 0))
                    lat = float(d.get("latency_ms_per_sample", np.nan))
                    if np.isfinite(lat):
                        total_latency_ms += lat
                    if i == 1:
                        single_latency = lat

            meta_sec = float(row.get("stack_time_sec", 0))
            meta_lat = float(row.get("meta_latency_ms_per_sample", 0))

            time_rows.append({
                "dataset": ds,
                "n_samples": int(row.get("n_train", 0)) + int(row.get("n_test", 0)),
                "k": k_target,
                "stack_base_sec": total_base_sec,
                "meta_sec": meta_sec,
                "total_sec": total_base_sec + meta_sec,
                "total_min": (total_base_sec + meta_sec) / 60.0,
            })

            latency_rows.append({
                "dataset": ds,
                "k": k_target,
                "single_model_ms": single_latency,
                "ensemble_ms": total_latency_ms + meta_lat,
            })

    if time_rows:
        df9 = pd.DataFrame(time_rows)
        # Average across folds
        df9_avg = df9.groupby("dataset").mean(numeric_only=True).reset_index()
        df9_avg.to_csv(os.path.join(save_dir, "table9_training_time.csv"), index=False)
        _p(f"  Saved table9")

    if latency_rows:
        df10 = pd.DataFrame(latency_rows)
        df10_avg = df10.groupby("dataset").mean(numeric_only=True).reset_index()
        # Add average row
        avg_row = df10_avg.mean(numeric_only=True)
        avg_row["dataset"] = "Average"
        df10_avg = pd.concat([df10_avg, pd.DataFrame([avg_row])], ignore_index=True)
        df10_avg.to_csv(os.path.join(save_dir, "table10_latency.csv"), index=False)
        _p(f"  Saved table10")


# =========================================================
# Table 11/12: Comparative Performance
# =========================================================
def make_table11_12(result_dirs, ml_dirs, save_dir, k_target=15):
    """Compare TabPFN-Anchor vs ML baselines"""
    _p("[Table 11/12] Comparative Performance")

    # TabPFN-Anchor results (average across folds at k=k_target)
    anchor_rows = []
    for rd in result_dirs:
        csv = os.path.join(rd, "metrics_by_k.csv")
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        sub = df[df["k"] == k_target]
        for _, row in sub.iterrows():
            anchor_rows.append(row.to_dict())

    # ML baseline results
    ml_rows = []
    for md in ml_dirs:
        for ds in Config.ALL_DATASETS:
            csv = os.path.join(md, f"{ds}_ml_baselines.csv")
            if os.path.exists(csv):
                df = pd.read_csv(csv)
                for _, row in df.iterrows():
                    ml_rows.append(row.to_dict())

    if not anchor_rows and not ml_rows:
        _p("  No results found")
        return

    # Aggregate TabPFN-Anchor across folds
    if anchor_rows:
        df_anchor = pd.DataFrame(anchor_rows)
        df_anchor_avg = df_anchor.groupby("dataset").agg({
            "precision_weighted": "mean",
            "recall_weighted": "mean",
            "f1_weighted": "mean",
            "auroc_weighted_ovr": "mean",
        }).reset_index()
        df_anchor_avg["model"] = "TabPFN-Anchor"
    else:
        df_anchor_avg = pd.DataFrame()

    # Aggregate ML baselines across folds
    if ml_rows:
        df_ml = pd.DataFrame(ml_rows)
        df_ml_avg = df_ml.groupby(["dataset", "model"]).agg({
            "precision_weighted": "mean",
            "recall_weighted": "mean",
            "f1_weighted": "mean",
            "auroc_weighted_ovr": "mean",
        }).reset_index()
    else:
        df_ml_avg = pd.DataFrame()

    # Combine
    df_combined = pd.concat([df_ml_avg, df_anchor_avg], ignore_index=True)

    if not df_combined.empty:
        # Table 11: Overall average
        t11 = df_combined.groupby("model").agg({
            "precision_weighted": "mean",
            "recall_weighted": "mean",
            "f1_weighted": "mean",
            "auroc_weighted_ovr": "mean",
        }).reset_index()
        t11.to_csv(os.path.join(save_dir, "table11_overall_comparison.csv"), index=False)
        _p(f"  Saved table11")

        # Table 12: Per-dataset F1
        t12 = df_combined.pivot_table(
            index="model", columns="dataset", values="f1_weighted",
        )
        t12["Average"] = t12.mean(axis=1)
        t12.to_csv(os.path.join(save_dir, "table12_per_dataset_f1.csv"))
        _p(f"  Saved table12")


# =========================================================
# Table 14: Top 10 Anchor Genes
# =========================================================
def make_table14(anchor_dirs, save_dir):
    """Top 10 anchor genes from SHAP ranking (averaged across folds)"""
    _p("[Table 14] Top 10 Anchor Genes")

    rows = []
    for ds in Config.ALL_DATASETS:
        # Collect top genes from each fold
        gene_scores = {}
        n_folds = 0
        for ad in anchor_dirs:
            csv = os.path.join(ad, ds, "global_shap_rank.csv")
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv)
            n_folds += 1
            for _, r in df.iterrows():
                feat = str(r["Feature"])
                imp = float(r.get("ShapImportance", 0))
                rank = int(r.get("Rank", 9999))
                if feat not in gene_scores:
                    gene_scores[feat] = {"imp_sum": 0, "rank_sum": 0, "count": 0}
                gene_scores[feat]["imp_sum"] += imp
                gene_scores[feat]["rank_sum"] += rank
                gene_scores[feat]["count"] += 1

        if not gene_scores:
            continue

        # Rank by average importance
        for g in gene_scores:
            gene_scores[g]["avg_imp"] = gene_scores[g]["imp_sum"] / gene_scores[g]["count"]
            gene_scores[g]["avg_rank"] = gene_scores[g]["rank_sum"] / gene_scores[g]["count"]

        sorted_genes = sorted(gene_scores.items(), key=lambda x: -x[1]["avg_imp"])
        top10 = [g[0] for g in sorted_genes[:10]]

        # Extract gene name from feature ID (e.g., N:GEXP::ESR1:2099: → ESR1)
        gene_names = []
        for f in top10:
            parts = f.split(":")
            # Try to find gene name (usually 4th element in N:GEXP::GENE:ID:)
            name = f
            for p in parts:
                if p and p != "N" and p != "GEXP" and not p.isdigit() and p != "?":
                    name = p
                    break
            gene_names.append(name)

        rows.append({
            "Dataset": ds,
            "Top_10_Anchor_Genes": ", ".join(gene_names),
        })
        _p(f"  {ds}: {', '.join(gene_names[:5])}...")

    if rows:
        df14 = pd.DataFrame(rows)
        df14.to_csv(os.path.join(save_dir, "table14_top10_anchors.csv"), index=False)
        _p(f"  Saved table14")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", default=[],
                        help="Stacking result dirs (one per fold)")
    parser.add_argument("--ml_dirs", nargs="+", default=[],
                        help="ML baseline dirs (one per fold)")
    parser.add_argument("--anchor_dirs", nargs="+", default=[],
                        help="SHAP anchor dirs (one per fold)")
    parser.add_argument("--save_dir", type=str, default="./paper_tables")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--k_target", type=int, default=Config.STACK_K)
    args = parser.parse_args()

    if args.data_dir:
        Config.DATA_SOURCE_DIR = Path(args.data_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    # Table 3/4: Dataset summary (always available from data)
    make_table3_4(args.save_dir)

    # Table 7: K convergence
    if args.result_dirs:
        make_table7(args.result_dirs, args.save_dir)

    # Table 9/10: Time & Latency
    if args.result_dirs:
        make_table9_10(args.result_dirs, args.save_dir, args.k_target)

    # Table 11/12: Comparative performance
    if args.result_dirs or args.ml_dirs:
        make_table11_12(args.result_dirs, args.ml_dirs, args.save_dir, args.k_target)

    # Table 14: Top anchor genes
    if args.anchor_dirs:
        make_table14(args.anchor_dirs, args.save_dir)

    _p(f"\n[Done] All tables saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
