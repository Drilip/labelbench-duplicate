#!/usr/bin/env bash
set -euo pipefail

N="${1:-3}"
REPO_DIR="/work/LabelBench/LabelBench"
LOG_DIR="/work/LabelBench/logs/randinit"
PYTHON_BIN="/work/LabelBench/env/bin/python"
mkdir -p "$LOG_DIR"

"$PYTHON_BIN" -c "import clip; clip.load('ViT-B/32', device='cpu')" >/dev/null 2>&1 || true

SEEDS=(1234 3334155 6667077 9999999)
STRATEGIES=(random confidence)
selected_seeds=("${SEEDS[@]:0:$N}")

# Remove any existing session to avoid stale windows.
tmux has-session -t randinit 2>/dev/null && tmux kill-session -t randinit || true

first_window=true
seed_index=0
for seed in "${selected_seeds[@]}"; do
  strategy_index=0
  for strategy in "${STRATEGIES[@]}"; do
    gpu=$(((seed_index + strategy_index) % 2))
    window_name="${strategy}_${seed}_gpu${gpu}"
    cmd="cd \"$REPO_DIR\" && export PYTHONPATH=\"$REPO_DIR\" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_RUN_GROUP=randinit WANDB_TAGS=randinit CUDA_VISIBLE_DEVICES=${gpu} && \"$PYTHON_BIN\" main.py --seed ${seed} --wandb_name ATDL-labelbench --dataset cifar10 --data_dir /work/LabelBench/data --metric multi_class --batch_size 1000 --num_batch 10 --embed_model_config none.json --classifier_model_config clip_ViTB32_randinit.json --strategy_config ${strategy}_sampling.json --trainer_config cifar10/flexmatch/clip_ViTB32_finetune.json --corrupter_config noiseless.json > ${LOG_DIR}/${window_name}.log 2>&1"
    if [[ "$first_window" == true ]]; then
      tmux new-session -d -s randinit -n "$window_name" "$cmd"
      first_window=false
    else
      tmux new-window -t randinit -n "$window_name" "$cmd"
    fi
    strategy_index=$((strategy_index + 1))
  done
  seed_index=$((seed_index + 1))
done

tmux list-windows -t randinit
