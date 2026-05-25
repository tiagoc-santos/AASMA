#!/usr/bin/env bash
set -euo pipefail

# Run from any location; this script assumes it is placed inside src/
# beside training.py.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TIMESTEPS="${TIMESTEPS:-3000000}"
LAYOUT="${LAYOUT:-three_chefs}"
ARCHITECTURE="${ARCHITECTURE:-cnn}"
NUM_CPU="${NUM_CPU:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
RESULTS_CSV="${RESULTS_CSV:-final_common_results.csv}"
RESET_RESULTS="${RESET_RESULTS:-false}"

# Final comparison: ad hoc curriculum versus curriculum/self-play baseline.
SEEDS=(42 101 202)
MODES=("adhoc_curriculum" "curriculum")

LOG_DIR="../logs/final_training"
mkdir -p "$LOG_DIR" "../models" "../heatmaps" "../gameplay_gifs"

if [[ "$RESET_RESULTS" == "true" ]]; then
    rm -f "../${RESULTS_CSV}"
    echo "Removed existing results file: ../${RESULTS_CSV}"
fi

echo "============================================================"
echo "Final experiment runner"
echo "Modes:        ${MODES[*]}"
echo "Seeds:        ${SEEDS[*]}"
echo "Timesteps:    ${TIMESTEPS}"
echo "Layout:       ${LAYOUT}"
echo "Architecture: ${ARCHITECTURE}"
echo "Eval episodes:${EVAL_EPISODES}"
echo "Results CSV:  ../${RESULTS_CSV}"
echo "============================================================"

for mode in "${MODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_name="${ARCHITECTURE}_${mode}_${TIMESTEPS}_seed${seed}"
        log_file="${LOG_DIR}/${run_name}.log"

        echo
        echo "============================================================"
        echo "Starting ${mode} | seed ${seed}"
        echo "Log: ${log_file}"
        echo "============================================================"

        # training.py trains from scratch because --model and
        # --resume_stage3_from are intentionally omitted.
        # It then evaluates the trained model on the common suite.
        "$PYTHON_BIN" training.py \
            --timesteps "$TIMESTEPS" \
            --train_partner_mode "$mode" \
            --architecture "$ARCHITECTURE" \
            --layout_name "$LAYOUT" \
            --num_cpu "$NUM_CPU" \
            --seed "$seed" \
            --eval_suite common \
            --eval_episodes "$EVAL_EPISODES" \
            --deterministic_ego true \
            --deterministic_partner false \
            --results_csv "$RESULTS_CSV" \
            2>&1 | tee "$log_file"

        echo "Completed ${mode} | seed ${seed}"
    done
done

echo
echo "============================================================"
echo "All final training/evaluation runs completed."
echo "Results: ../${RESULTS_CSV}"
echo "Logs:    ${LOG_DIR}/"
echo "============================================================"
