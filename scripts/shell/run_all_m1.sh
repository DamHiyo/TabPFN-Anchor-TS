#!/bin/bash
# Paper3: TabPFN M1 — 3가지 feature selection 방식 모두 실행
#
# 실행:
#   cd /data2/project/2026winter/jud9679/paper3/scripts
#   nohup bash run_all_m1.sh > ../logs/tabpfn_m1.log 2>&1 &
set -e

cd "$(dirname "$0")"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] TabPFN M1 전체 시작"

# SHAP_RF (LR→RF→SHAP pool)
echo ""
echo "===== TabPFN M1: SHAP_RF ====="
python run_tabpfn_m1.py --pool_method SHAP_RF --device cuda:0
echo "[SHAP_RF] DONE"

# SHAP_MI (LR→MI→SHAP pool)
echo ""
echo "===== TabPFN M1: SHAP_MI ====="
python run_tabpfn_m1.py --pool_method SHAP_MI --device cuda:0
echo "[SHAP_MI] DONE"

# SHAP_EN (EN→SHAP pool)
echo ""
echo "===== TabPFN M1: SHAP_EN ====="
python run_tabpfn_m1.py --pool_method SHAP_EN --device cuda:0
echo "[SHAP_EN] DONE"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] TabPFN M1 전체 완료"
echo "Results:"
echo "  results/tabpfn/m1_shap_rf/"
echo "  results/tabpfn/m1_shap_mi/"
echo "  results/tabpfn/m1_shap_en/"
