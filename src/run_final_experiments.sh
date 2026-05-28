#!/usr/bin/env bash
set -euo pipefail

# Place this script inside src/ beside training.py.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TIMESTEPS="${TIMESTEPS:-3000000}"
LAYOUT="${LAYOUT:-three_chefs}"
ARCHITECTURE="${ARCHITECTURE:-rnn}"
NUM_CPU="${NUM_CPU:-4}"
EVAL_SUITE="${EVAL_SUITE:-test}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
RESULTS_CSV="${RESULTS_CSV:-selfplay_results.csv}"
RESET_RESULTS="${RESET_RESULTS:-false}"
ADHOC_MODE="${ADHOC_MODE:-adhoc_curriculum}"
SELF_PLAY_MODE="${SELF_PLAY_MODE:-self_play}"

SELF_PLAY_SEEDS=(101 202)

LOG_DIR="../logs/final_training"
mkdir -p "$LOG_DIR" "../models" "../heatmaps" "../gameplay_gifs"

if [[ "$RESET_RESULTS" == "true" ]]; then
    rm -f "../${RESULTS_CSV}"
    echo "Removed existing results file: ../${RESULTS_CSV}"
fi

run_experiment() {
    local mode="$1"
    local seed="$2"
    local run_name="${ARCHITECTURE}_${mode}_${TIMESTEPS}_seed${seed}"
    local log_file="${LOG_DIR}/${run_name}.log"

    echo
    echo "============================================================"
    echo "Starting ${mode} | seed ${seed}"
    echo "Log: ${log_file}"
    echo "============================================================"

    "$PYTHON_BIN" training.py \
        --timesteps "$TIMESTEPS" \
        --train_partner_mode "$mode" \
        --architecture "$ARCHITECTURE" \
        --layout_name "$LAYOUT" \
        --num_cpu "$NUM_CPU" \
        --seed "$seed" \
        --eval_suite "$EVAL_SUITE" \
        --eval_episodes "$EVAL_EPISODES" \
        --deterministic_ego true \
        --deterministic_partner false \
        --results_csv "$RESULTS_CSV" \
        2>&1 | tee "$log_file"

    echo "Completed ${mode} | seed ${seed}"
}

echo "============================================================"
echo "Remaining final experiment runner"
echo "Self-play mode:  ${SELF_PLAY_MODE}"
echo "Self-play seeds: ${SELF_PLAY_SEEDS[*]}"
echo "Timesteps:       ${TIMESTEPS}"
echo "Layout:          ${LAYOUT}"
echo "Architecture:    ${ARCHITECTURE}"
echo "Eval suite:      ${EVAL_SUITE}"
echo "Eval episodes:   ${EVAL_EPISODES}"
echo "Results CSV:     ../${RESULTS_CSV}"
echo "============================================================"

for seed in "${SELF_PLAY_SEEDS[@]}"; do
    run_experiment "$SELF_PLAY_MODE" "$seed"
done

echo
echo "============================================================"
echo "All requested training/evaluation runs completed."
echo "Results: ../${RESULTS_CSV}"
echo "Logs:    ${LOG_DIR}/"
echo "============================================================"
