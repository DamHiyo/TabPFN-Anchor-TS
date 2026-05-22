# -*- coding: utf-8 -*-
"""
step5_ml_baselines.py

ML Baselines: SVM, Random Forest, XGBoost, MLP
- 각 fold의 train/test split에서 평가
- Grid search hyperparameter tuning (StratifiedKFold inner CV)
- 출력: {save_dir}/{fold_name}/{dataset}_ml_baselines.csv

실행 예시:
  python step5_ml_baselines.py \
    --datasets BRCA COADREAD GEA KIRCKICH SKCM THCA UCEC BLCA HNSC LGGGBM LUAD LUSC \
    --fold_col R1:F1 \
    --save_dir ./final_results/ml_baselines/R1_F1
"""

import os
import sys
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, log_loss,
)

from config import Config
from data_loader import DataLoader

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


def _p(msg: str):
    print(msg, flush=True)


def _compute_metrics(y_true, y_pred, y_proba, n_classes):
    met = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if y_proba is not None:
        try:
            met["auroc_weighted_ovr"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            met["auroc_weighted_ovr"] = np.nan
        try:
            met["logloss"] = float(log_loss(y_true, y_proba, labels=np.arange(n_classes)))
        except Exception:
            met["logloss"] = np.nan
    return met


# Model definitions with grid search params
def get_models_and_params(n_classes, seed):
    models = {}

    # SVM (Linear)
    models["SVM"] = {
        "estimator": SVC(kernel="linear", probability=True, random_state=seed),
        "param_grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    }

    # Random Forest
    models["RF"] = {
        "estimator": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "param_grid": {
            "n_estimators": [500, 1000],
            "max_depth": [None, 20, 50],
        },
    }

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = {
            "estimator": XGBClassifier(
                random_state=seed, use_label_encoder=False,
                eval_metric="mlogloss", n_jobs=-1, tree_method="hist",
            ),
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            },
        }

    # LightGBM
    if HAS_LGBM:
        models["LightGBM"] = {
            "estimator": LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1),
            "param_grid": {
                "n_estimators": [100, 300],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            },
        }

    # CatBoost
    if HAS_CAT:
        models["CatBoost"] = {
            "estimator": CatBoostClassifier(random_state=seed, verbose=0),
            "param_grid": {
                "iterations": [100, 300],
                "depth": [4, 6],
                "learning_rate": [0.01, 0.1],
            },
        }

    # GradientBoosting
    models["GradientBoosting"] = {
        "estimator": GradientBoostingClassifier(random_state=seed),
        "param_grid": {
            "n_estimators": [100, 300],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
        },
    }

    # AdaBoost
    models["AdaBoost"] = {
        "estimator": AdaBoostClassifier(random_state=seed),
        "param_grid": {
            "n_estimators": [100, 300],
            "learning_rate": [0.01, 0.1, 1.0],
        },
    }

    # ExtraTrees
    models["ExtraTrees"] = {
        "estimator": ExtraTreesClassifier(random_state=seed, n_jobs=-1),
        "param_grid": {
            "n_estimators": [500, 1000],
            "max_depth": [None, 20, 50],
        },
    }

    # SVM_RBF
    models["SVM_RBF"] = {
        "estimator": SVC(kernel="rbf", probability=True, random_state=seed),
        "param_grid": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
    }

    # LogisticReg
    models["LogisticReg"] = {
        "estimator": LogisticRegression(max_iter=5000, random_state=seed, n_jobs=-1),
        "param_grid": {"C": [0.01, 0.1, 1.0, 10.0]},
    }

    # MLP
    models["MLP"] = {
        "estimator": MLPClassifier(random_state=seed, max_iter=500, early_stopping=True),
        "param_grid": {
            "hidden_layer_sizes": [(256,), (256, 128), (512, 256)],
            "alpha": [1e-4, 1e-3],
            "learning_rate_init": [1e-3, 1e-4],
        },
    }

    return models


def run_one_dataset(dataset, fold_col, save_dir, seed):
    _p(f"\n{'='*50}")
    _p(f"[ML Baselines] {dataset} (fold={fold_col})")

    loader = DataLoader(dataset)
    X, y, cv_df = loader.load_data()
    if X is None:
        return

    # Train/test split
    mask_train = (cv_df[fold_col].values == 0)
    mask_test = (cv_df[fold_col].values == 1)

    le = LabelEncoder()
    y_all = le.fit_transform(y.astype(str))
    n_classes = len(le.classes_)

    X_train = X.loc[mask_train].values
    X_test = X.loc[mask_test].values
    y_train = y_all[mask_train]
    y_test = y_all[mask_test]

    # StandardScaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    _p(f"  Train: {X_train_sc.shape}, Test: {X_test_sc.shape}, Classes: {n_classes}")

    models = get_models_and_params(n_classes, seed)
    results = []

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    for name, spec in models.items():
        _p(f"  [{name}] Grid search...")
        t0 = time.time()

        gs = GridSearchCV(
            spec["estimator"], spec["param_grid"],
            cv=inner_cv, scoring="f1_weighted",
            n_jobs=-1, refit=True,
        )
        gs.fit(X_train_sc, y_train)
        train_time = time.time() - t0

        # Predict
        t0 = time.time()
        y_pred = gs.predict(X_test_sc)
        infer_time = time.time() - t0

        y_proba = None
        if hasattr(gs, "predict_proba"):
            try:
                y_proba = gs.predict_proba(X_test_sc)
                # Align proba columns
                if y_proba.shape[1] < n_classes:
                    aligned = np.zeros((y_proba.shape[0], n_classes), dtype=np.float32)
                    for j, c in enumerate(gs.classes_):
                        aligned[:, int(c)] = y_proba[:, j]
                    y_proba = aligned
            except Exception:
                pass

        met = _compute_metrics(y_test, y_pred, y_proba, n_classes)
        met["model"] = name
        met["dataset"] = dataset
        met["fold_col"] = fold_col
        met["best_params"] = str(gs.best_params_)
        met["train_time_sec"] = train_time
        met["infer_time_sec"] = infer_time
        met["infer_time_ms_per_sample"] = (infer_time * 1000.0) / max(1, X_test_sc.shape[0])
        met["n_train"] = X_train_sc.shape[0]
        met["n_test"] = X_test_sc.shape[0]
        met["n_classes"] = n_classes

        results.append(met)
        _p(f"  [{name}] f1={met['f1_weighted']:.4f} time={train_time:.1f}s params={gs.best_params_}")

    df = pd.DataFrame(results)
    save_path = os.path.join(save_dir, f"{dataset}_ml_baselines.csv")
    df.to_csv(save_path, index=False)
    _p(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=Config.ALL_DATASETS)
    parser.add_argument("--fold_col", type=str, default="R1:F1")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=Config.SEED)
    args = parser.parse_args()

    if args.data_dir:
        Config.DATA_SOURCE_DIR = Path(args.data_dir)

    fold_tag = args.fold_col.replace(":", "")
    if args.save_dir is None:
        args.save_dir = str(Config.RESULT_DIR / "ml_baselines" / fold_tag)

    os.makedirs(args.save_dir, exist_ok=True)

    for i, ds in enumerate(args.datasets, start=1):
        _p(f"\n[{i}/{len(args.datasets)}] {ds}")
        try:
            run_one_dataset(ds, args.fold_col, args.save_dir, args.seed)
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
