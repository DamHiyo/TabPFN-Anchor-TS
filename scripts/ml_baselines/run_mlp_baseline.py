"""
MLP Baseline: sklearn MLPClassifier on all ~20K features.
12 datasets x 5 folds, results saved to CSV.
"""
import sys, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)

sys.path.insert(0, "/data2/project/2026winter/jud9679/paper2/move")
from config import Config
from data_loader import DataLoader

RESULT_DIR = Path("/data2/project/2026winter/jud9679/paper3/results/dl_baselines")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULT_DIR / "mlp_results.csv"

DATASETS = sorted(['BRCA','COADREAD','GEA','KIRCKICH','SKCM','THCA','UCEC',
                    'BLCA','HNSC','LGGGBM','LUAD','LUSC'])

# Grid search over these
MLP_CONFIGS = [
    {"hidden_layer_sizes": (256, 128), "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (256, 128), "learning_rate_init": 0.0005},
    {"hidden_layer_sizes": (512, 256), "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (512, 256, 128), "learning_rate_init": 0.001},
    {"hidden_layer_sizes": (128, 64), "learning_rate_init": 0.001},
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_dataset(dataset):
    loader = DataLoader(dataset)
    X, y, cv_df = loader.load_data()
    if X is None:
        return

    fold_cols = [c for c in cv_df.columns if c.startswith("R1:F")]
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    is_multi = len(le.classes_) > 2
    X_np = X.values

    rows = []
    for fold_col in fold_cols:
        train_idx = (pd.to_numeric(cv_df[fold_col], errors="coerce") == 0).values
        test_idx  = (pd.to_numeric(cv_df[fold_col], errors="coerce") == 1).values
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        best_f1, best_row = -1, None
        for cfg in MLP_CONFIGS:
            model = MLPClassifier(
                hidden_layer_sizes=cfg["hidden_layer_sizes"],
                learning_rate_init=cfg["learning_rate_init"],
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=42,
                batch_size=min(256, X_tr.shape[0]),
            )
            start = time.time()
            model.fit(X_tr_sc, y_tr)
            elapsed = time.time() - start

            pred = model.predict(X_te_sc)
            proba = model.predict_proba(X_te_sc)
            f1w = f1_score(y_te, pred, average="weighted", zero_division=0)

            if f1w > best_f1:
                best_f1 = f1w
                acc = accuracy_score(y_te, pred)
                f1_macro = f1_score(y_te, pred, average="macro", zero_division=0)
                prec_w = precision_score(y_te, pred, average="weighted", zero_division=0)
                rec_w = recall_score(y_te, pred, average="weighted", zero_division=0)
                prec_m = precision_score(y_te, pred, average="macro", zero_division=0)
                rec_m = recall_score(y_te, pred, average="macro", zero_division=0)
                try:
                    auc = roc_auc_score(
                        y_te, proba if is_multi else proba[:, 1],
                        multi_class="ovr" if is_multi else None
                    )
                except:
                    auc = np.nan
                best_row = {
                    "Dataset": dataset, "Fold": fold_col, "Model": "MLP",
                    "Config": str(cfg["hidden_layer_sizes"]),
                    "LR": cfg["learning_rate_init"],
                    "ACC": acc, "AUC": auc,
                    "F1_Weighted": f1w, "F1_Macro": f1_macro,
                    "Prec_Weighted": prec_w, "Rec_Weighted": rec_w,
                    "Prec_Macro": prec_m, "Rec_Macro": rec_m,
                    "Time": elapsed,
                }
            del model; gc.collect()

        log(f"[{dataset}] {fold_col} MLP best F1={best_f1:.4f} cfg={best_row['Config']}")
        rows.append(best_row)

    df_out = pd.DataFrame(rows)
    header = not OUT_CSV.exists()
    df_out.to_csv(OUT_CSV, mode="a", index=False, header=header)
    log(f"[{dataset}] DONE")


def main():
    log(f"=== MLP Baseline | {len(DATASETS)} datasets ===")
    for i, ds in enumerate(DATASETS, 1):
        log(f"[{i}/{len(DATASETS)}] {ds}")
        try:
            run_dataset(ds)
        except Exception as e:
            log(f"[{ds}] ERROR: {e}")
            import traceback; traceback.print_exc()
    log("=== ALL DONE ===")


if __name__ == "__main__":
    main()
