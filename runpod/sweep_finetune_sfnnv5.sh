#!/usr/bin/env bash
set -euo pipefail

EPOCH_SIZE="${EPOCH_SIZE:-50000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NETWORK_SAVE_PERIOD="${NETWORK_SAVE_PERIOD:-5}"
SAVE_LAST_NETWORK="${SAVE_LAST_NETWORK:-False}"

run_one() {
  local run_name="$1"
  local lr="$2"
  local epochs="$3"
  local lambda="$4"

  echo "=== $run_name | LR=$lr | MAX_EPOCHS=$epochs | LAMBDA=$lambda ==="
  RUN_NAME="$run_name" \
  LR="$lr" \
  MAX_EPOCHS="$epochs" \
  LAMBDA="$lambda" \
  EPOCH_SIZE="$EPOCH_SIZE" \
  BATCH_SIZE="$BATCH_SIZE" \
  NETWORK_SAVE_PERIOD="$NETWORK_SAVE_PERIOD" \
  SAVE_LAST_NETWORK="$SAVE_LAST_NETWORK" \
  bash runpod/finetune_sfnnv5_from_default.sh
}

# Sweep around the current best: LR=1e-5, 5 epochs, lambda=0.2.
run_one ft_lr5e-6_e5_l02 5e-6 5 0.2
run_one ft_lr1e-5_e8_l02 1e-5 8 0.2
run_one ft_lr2e-5_e5_l02 2e-5 5 0.2
run_one ft_lr1e-5_e5_l01 1e-5 5 0.1
run_one ft_lr1e-5_e5_l04 1e-5 5 0.4
