#!/usr/bin/env bash
# Reproduce the CIFAR-10 selection-via-proxy experiments from LabelBench v4:
#   Figure 5a: shallow proxy, 1,000 labels/round, 10 rounds
#   Figure 5b: shallow proxy,   200 labels/round, 20 rounds (4,000 labels)
#   Table 6:   10,000-label endpoints for fine-tuning, shallow, and linear
#   Smoke test: one real two-round proxy run plus one target evaluation
#
# Examples:
#   WANDB_NAME=my_entity ./example_run.sh             # Figure 5a + 5b
#   ./example_run.sh all --dry-run
#   WANDB_NAME=my_entity ./example_run.sh --smoke-test
#   WANDB_NAME=my_entity ./example_run.sh figure5a
#   WANDB_NAME=my_entity GPU_MASKS=0 ./example_run.sh all
#
# Results are logged to Weights & Biases. The launchers deterministically use
# seeds 1234, 3334155, 6667077, and 9999999 when NUM_RUNS=4.
# Start with one GPU on a clean data directory: concurrent first-time jobs can
# race while writing the shared CLIP embedding cache. Multiple GPUs are safe
# after the cache under DATA_DIR/cifar10 has been populated.

set -euo pipefail

MODE="fig5"
DRY_RUN=false
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=.venv/bin/python
else
  PYTHON_BIN=python3
fi
DATA_DIR="${DATA_DIR:-./data}"
NUM_RUNS="${NUM_RUNS:-4}"
RUN_PER_DEVICE="${RUN_PER_DEVICE:-1}"
read -r -a GPU_MASK_ARRAY <<< "${GPU_MASKS:-0}"

for arg in "$@"; do
  case "${arg}" in
    fig5|figure5a|figure5b|table6|all|smoke)
      MODE="${arg}"
      ;;
    --smoke-test)
      MODE="smoke"
      ;;
    --dry-run)
      DRY_RUN=true
      ;;
    *)
      echo "Usage: $0 {fig5|figure5a|figure5b|table6|all|--smoke-test} [--dry-run]" >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" == true ]]; then
  WANDB_NAME="${WANDB_NAME:-dry-run-entity}"
  LAUNCHER_DRY_RUN_ARGS=(--dry_run)
  echo "Dry run: commands will be validated and printed, but no child jobs will start."
else
  : "${WANDB_NAME:?Set WANDB_NAME to your Weights & Biases entity/team name}"
  LAUNCHER_DRY_RUN_ARGS=()
fi

# Figure 5a and Table 6 contain BAIT. The checked-in Figure 5b artifacts use
# the other seven methods only.
STRATEGIES_1000=(
  random_sampling.json
  confidence_sampling.json
  entropy_sampling.json
  margin_sampling.json
  coreset_sampling.json
  galaxy_sampling.json
  badge_sampling.json
  bait_sampling.json
)
STRATEGIES_200=(
  random_sampling.json
  confidence_sampling.json
  entropy_sampling.json
  margin_sampling.json
  coreset_sampling.json
  galaxy_sampling.json
  badge_sampling.json
)

run_selection() {
  local batch_size="$1"
  local num_batches="$2"
  local classifier_config="$3"
  local trainer_config="$4"
  shift 4

  "${PYTHON_BIN}" mp_launcher.py \
    --wandb_name "${WANDB_NAME}" \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --metric multi_class \
    --batch_size "${batch_size}" \
    --num_batch "${num_batches}" \
    --embed_model_config clip_ViTB32.json \
    --classifier_model_config "${classifier_config}" \
    --trainer_config "${trainer_config}" \
    --corrupter_config noiseless.json \
    --strategies "$@" \
    --num_runs "${NUM_RUNS}" \
    --run_per_device "${RUN_PER_DEVICE}" \
    --gpu_masks "${GPU_MASK_ARRAY[@]}" \
    "${LAUNCHER_DRY_RUN_ARGS[@]}"
}

run_proxy_evaluation() {
  local selection_batch_size="$1"
  local proxy_name="$2"
  local eval_batch_size="$3"
  local eval_num_batches="$4"
  shift 4

  # point_evaluation.py uses substrings (without .json) to locate the W&B run
  # or the checked-in selection pickle.
  local strategy_names=()
  local strategy
  for strategy in "$@"; do
    strategy_names+=("${strategy%.json}")
  done

  "${PYTHON_BIN}" mp_eval_launcher.py \
    --wandb_name "${WANDB_NAME}" \
    --wandb_project "Active Learning, cifar10, Batch Size=${selection_batch_size}" \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --embed_model_config clip_ViTB32 \
    --classifier_model_config "${proxy_name}" \
    --strategies "${strategy_names[@]}" \
    --trainer_config flexmatch \
    --corrupter_config noiseless.json \
    --eval_metric multi_class \
    --eval_batch_size "${eval_batch_size}" \
    --eval_num_batch "${eval_num_batches}" \
    --eval_classifier_model_config clip_ViTB32_pretrained.json \
    --eval_trainer_config cifar10/flexmatch/clip_ViTB32_finetune.json \
    --num_runs "${NUM_RUNS}" \
    --run_per_device "${RUN_PER_DEVICE}" \
    --gpu_masks "${GPU_MASK_ARRAY[@]}" \
    "${LAUNCHER_DRY_RUN_ARGS[@]}"
}

run_figure5a() {
  run_selection 1000 10 shallow.json \
    cifar10/flexmatch/clip_ViTB32_shallow.json "${STRATEGIES_1000[@]}"
  run_proxy_evaluation 1000 shallow 1000 10 "${STRATEGIES_1000[@]}"
}

run_figure5b() {
  run_selection 200 20 shallow.json \
    cifar10/flexmatch/clip_ViTB32_shallow.json "${STRATEGIES_200[@]}"
  run_proxy_evaluation 200 shallow 200 20 "${STRATEGIES_200[@]}"
}

run_figure5() {
  run_figure5a
  run_figure5b
}

run_finetune_table6() {
  "${PYTHON_BIN}" mp_launcher.py \
    --wandb_name "${WANDB_NAME}" \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --metric multi_class \
    --batch_size 1000 \
    --num_batch 10 \
    --embed_model_config none.json \
    --classifier_model_config clip_ViTB32_pretrained.json \
    --trainer_config cifar10/flexmatch/clip_ViTB32_finetune.json \
    --corrupter_config noiseless.json \
    --strategies "${STRATEGIES_1000[@]}" \
    --num_runs "${NUM_RUNS}" \
    --run_per_device "${RUN_PER_DEVICE}" \
    --gpu_masks "${GPU_MASK_ARRAY[@]}" \
    "${LAUNCHER_DRY_RUN_ARGS[@]}"
}

run_table6() {
  run_finetune_table6

  run_selection 1000 10 shallow.json \
    cifar10/flexmatch/clip_ViTB32_shallow.json "${STRATEGIES_1000[@]}"
  run_proxy_evaluation 1000 shallow 10000 1 "${STRATEGIES_1000[@]}"

  run_selection 1000 10 linear.json \
    cifar10/flexmatch/clip_ViTB32_linear.json "${STRATEGIES_1000[@]}"
  run_proxy_evaluation 1000 linear 10000 1 "${STRATEGIES_1000[@]}"
}

run_smoke_test() {
  # This is a real end-to-end run, but deliberately tiny: one seed, Margin,
  # two 32-label proxy rounds, and one target evaluation on all 64 labels.
  # The smoke configs use one augmentation seed and one training epoch.
  "${PYTHON_BIN}" mp_launcher.py \
    --wandb_name "${WANDB_NAME}" \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --metric multi_class \
    --batch_size 32 \
    --num_batch 2 \
    --embed_model_config clip_ViTB32_smoke.json \
    --classifier_model_config shallow.json \
    --trainer_config cifar10/flexmatch/clip_ViTB32_shallow_smoke.json \
    --corrupter_config noiseless.json \
    --strategies margin_sampling.json \
    --num_runs 1 \
    --run_per_device 1 \
    --gpu_masks "${GPU_MASK_ARRAY[@]}" \
    "${LAUNCHER_DRY_RUN_ARGS[@]}"

  "${PYTHON_BIN}" mp_eval_launcher.py \
    --wandb_name "${WANDB_NAME}" \
    --wandb_project "Active Learning, cifar10, Batch Size=32" \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --embed_model_config clip_ViTB32_smoke \
    --classifier_model_config shallow \
    --strategies margin_sampling \
    --trainer_config shallow_smoke \
    --corrupter_config noiseless.json \
    --eval_metric multi_class \
    --eval_batch_size 64 \
    --eval_num_batch 1 \
    --eval_classifier_model_config clip_ViTB32_pretrained.json \
    --eval_trainer_config cifar10/flexmatch/clip_ViTB32_finetune_smoke.json \
    --num_runs 1 \
    --run_per_device 1 \
    --gpu_masks "${GPU_MASK_ARRAY[@]}" \
    "${LAUNCHER_DRY_RUN_ARGS[@]}"
}

case "${MODE}" in
  fig5)
    run_figure5
    ;;
  figure5a)
    run_figure5a
    ;;
  figure5b)
    run_figure5b
    ;;
  table6)
    run_table6
    ;;
  smoke)
    run_smoke_test
    ;;
  all)
    # Figure 5a's final point is also the shallow-network column of Table 6,
    # so this avoids repeating those shallow-proxy evaluations.
    run_finetune_table6
    run_figure5a
    run_selection 1000 10 linear.json \
      cifar10/flexmatch/clip_ViTB32_linear.json "${STRATEGIES_1000[@]}"
    run_proxy_evaluation 1000 linear 10000 1 "${STRATEGIES_1000[@]}"
    run_figure5b
    ;;
esac

if [[ "${DRY_RUN}" == true ]]; then
  echo "Dry run completed successfully; no training or evaluation jobs were started."
fi
