# -*- coding: utf-8 -*-
"""
data_loader.py

[Data Loader Module]
- TSV 파일을 읽어서 GEXP 피쳐(X)와 Labels(y)를 분리합니다.
- CV Fold 정보도 함께 로드하며, 샘플 ID 정렬(Alignment)을 수행합니다.
"""
import pandas as pd
from config import Config


class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.data_path = Config.get_data_path(dataset_name)
        self.cv_path = Config.get_cv_path(dataset_name)

    def load_data(self):
        """
        Returns:
            X (pd.DataFrame): GEXP Features (Samples x Features)
            y (pd.Series): Target Labels (Samples,)
            cv_df (pd.DataFrame): 5-Fold Indices
        """
        print(f"[{self.dataset_name}] Loading raw data...")
        try:
            df = pd.read_csv(self.data_path, sep="\t", index_col=0)
        except FileNotFoundError:
            print(f"[Error] File not found: {self.data_path}")
            return None, None, None

        # target column 결정
        if "Labels" in df.columns:
            target_col = "Labels"
        elif "Label" in df.columns:
            target_col = "Label"
        else:
            non_gexp = [c for c in df.columns if Config.FEATURE_TAG not in c]
            if not non_gexp:
                print("[Error] Target column not found (no non-GEXP columns).")
                return None, None, None
            target_col = non_gexp[0]
            print(f"[Warning] 'Labels/Label' not found. Using '{target_col}' as target.")

        y = df[target_col].copy()

        # feature columns (GEXP tag)
        feature_cols = [c for c in df.columns if Config.FEATURE_TAG in c]
        if len(feature_cols) == 0:
            print(f"[Error] No features found with tag '{Config.FEATURE_TAG}'.")
            return None, None, None

        X = df[feature_cols].copy()

        # CV 파일 로드
        try:
            cv_df = pd.read_csv(self.cv_path, sep="\t", index_col=0)
        except FileNotFoundError:
            print(f"[Error] CV file not found: {self.cv_path}")
            return None, None, None

        # Alignment (교집합)
        common = X.index.intersection(cv_df.index)
        if len(common) == 0:
            print("[Error] No common sample IDs between data and CV file.")
            return None, None, None

        if len(common) != len(X):
            print(f"[Info] Aligning samples: {len(X)} -> {len(common)} (Common with CV)")

        X = X.loc[common]
        y = y.loc[common]
        cv_df = cv_df.loc[common]

        print(f"[{self.dataset_name}] Ready.")
        print(f"   - Samples: {X.shape[0]}")
        print(f"   - Features: {X.shape[1]} (Tag: '{Config.FEATURE_TAG}')")
        print(f"   - Classes: {y.nunique()}")

        return X, y, cv_df


if __name__ == "__main__":
    Config.makedirs()
    loader = DataLoader("BLCA")
    X, y, cv = loader.load_data()
    if X is not None:
        print("[OK] DataLoader test passed.")
