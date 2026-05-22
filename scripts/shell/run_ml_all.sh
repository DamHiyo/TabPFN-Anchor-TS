#!/bin/bash
# Paper3: 12암종 × 5fold × ML models (전체 feature 사용)
#
# 실행:
#   cd /data2/project/2026winter/jud9679/paper3/scripts
#   nohup bash run_ml_all.sh > ../logs/ml_all.log 2>&1 &
set -e

cd "$(dirname "$0")"  # scripts/ 폴더로 이동

SAVE_BASE="../results/ml"
DATASETS="BRCA COADREAD GEA KIRCKICH SKCM THCA UCEC BLCA HNSC LGGGBM LUAD LUSC"
FOLDS=("R1:F1" "R1:F2" "R1:F3" "R1:F4" "R1:F5")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Paper3 ML 전체 시작"

for FOLD in "${FOLDS[@]}"; do
    FOLD_TAG="${FOLD//:/}"
    SAVE_DIR="${SAVE_BASE}/${FOLD_TAG}"
    mkdir -p "$SAVE_DIR"
    echo ""
    echo "===== ${FOLD} ====="

    python step5_ml_baselines.py \
        --datasets $DATASETS \
        --fold_col "$FOLD" \
        --save_dir "$SAVE_DIR"

    echo "[${FOLD}] DONE"
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Paper3 ML 완료"
echo "Results: ${SAVE_BASE}/"
