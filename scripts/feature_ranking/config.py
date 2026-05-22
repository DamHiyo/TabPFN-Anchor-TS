# -*- coding: utf-8 -*-
"""
config.py

[Project Configuration]
- 모든 경로와 실험 파라미터를 한곳에서 관리합니다.
- (중요) WORK_DIR / DATA_SOURCE_DIR 만 내 환경에 맞게 수정하면 됩니다.
"""
from pathlib import Path


class Config:
    # ==========================================
    # 1. Path Settings (내 환경에 맞게 수정)
    # ==========================================
    WORK_DIR = Path("/data2/project/2026winter/jud9679/paper3")
    DATA_SOURCE_DIR = WORK_DIR / "data"

    # 결과/로그/아티팩트 저장 경로
    RESULT_DIR = WORK_DIR / "results"
    LOG_DIR = WORK_DIR / "logs"
    ARTIFACT_DIR = WORK_DIR / "results" / "features"

    # ==========================================
    # 2. Data file name rules
    # ==========================================
    FILE_SUFFIX_DATA = "_v12_20210228.tsv"
    FILE_SUFFIX_CV = "_CVfolds_5FOLD_v12_20210228.tsv"

    # ==========================================
    # 3. Experiment Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    FEATURE_TAG = "GEXP"

    # ==========================================
    # 4. Pipeline Parameters (논문 기준)
    # ==========================================
    # Step 1-1: LR Permutation Importance
    LR_TOP_N = 6000          # P → 6000
    LR_PERM_REPEATS = 50
    LR_PERM_SCORING = "neg_log_loss"

    # Step 1-2: Random Forest
    RF_TOP_N = 500            # 2000 → 500
    RF_N_ESTIMATORS = 2000

    # Step 1-3: TabPFN + SHAP
    ANCHOR_N = 100            # 500 → 100 (anchor genes)
    SHAP_N_ESTIMATORS = 8     # SHAP ranking 용
    SHAP_BLOCK_SIZE = 500
    SHAP_FIT_MAX_SAMPLES = 1024
    SHAP_TEST_MAX_SAMPLES = 100

    # Step 2: Stacking
    STACK_N_ESTIMATORS = 32   # stacking 용 TabPFN
    BLOCK_SIZE = 500          # anchor(100) + residual(400)
    RESIDUAL_SIZE = 400
    STACK_K = 15              # 기본 스택 수

    # ==========================================
    # 5. Target Datasets (논문 12개 암종)
    # ==========================================
    ALL_DATASETS = [
        "BRCA", "COADREAD", "GEA", "KIRCKICH",
        "SKCM", "THCA", "UCEC", "BLCA",
        "HNSC", "LGGGBM", "LUAD", "LUSC",
    ]

    # CV fold columns
    FOLD_COLS = ["R1:F1", "R1:F2", "R1:F3", "R1:F4", "R1:F5"]

    @staticmethod
    def makedirs():
        Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        Config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_data_path(dataset: str) -> Path:
        return Config.DATA_SOURCE_DIR / f"{dataset}{Config.FILE_SUFFIX_DATA}"

    @staticmethod
    def get_cv_path(dataset: str) -> Path:
        return Config.DATA_SOURCE_DIR / f"{dataset}{Config.FILE_SUFFIX_CV}"


if __name__ == "__main__":
    Config.makedirs()
    print(f"[Config] WORK_DIR: {Config.WORK_DIR}")
    print(f"[Config] DATA_SOURCE_DIR: {Config.DATA_SOURCE_DIR}")
    print(f"[Config] RESULT_DIR: {Config.RESULT_DIR}")
