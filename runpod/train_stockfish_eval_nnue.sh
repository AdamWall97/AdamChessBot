#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${RUNPOD_DATA_DIR:-/workspace/chess_v4/data/goingawall1}"
WORK_DIR="${RUNPOD_WORK_DIR:-/workspace/work}"
NNUE_DIR="${NNUE_DIR:-$WORK_DIR/nnue-pytorch}"
STOCKFISH_BIN="${STOCKFISH_BIN:-$WORK_DIR/Stockfish-tools/src/stockfish}"

PLAIN_FILE="${PLAIN_FILE:-$DATA_DIR/nnue_plain.plain}"
DATASET_DIR="${DATASET_DIR:-$WORK_DIR/datasets}"
BINPACK_FILE="${BINPACK_FILE:-$DATASET_DIR/personal.binpack}"
RUN_DIR="${RUN_DIR:-$WORK_DIR/runs/goingawall1_eval}"
NET_DIR="${NET_DIR:-$WORK_DIR/nets}"
OUT_NNUE="${OUT_NNUE:-$NET_DIR/goingawall1_eval.nnue}"

MAX_EPOCHS="${MAX_EPOCHS:-40}"
EPOCH_SIZE="${EPOCH_SIZE:-200000}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
NETWORK_SAVE_PERIOD="${NETWORK_SAVE_PERIOD:-5}"
GPUS="${GPUS:-0}"

if [ ! -f "$PLAIN_FILE" ]; then
  echo "Missing plain training file: $PLAIN_FILE" >&2
  exit 1
fi

if [ ! -x "$STOCKFISH_BIN" ]; then
  echo "Missing Stockfish tools binary: $STOCKFISH_BIN" >&2
  echo "Run: bash runpod/bootstrap_nnue_tools.sh" >&2
  exit 1
fi

if [ ! -d "$NNUE_DIR" ]; then
  echo "Missing nnue-pytorch checkout: $NNUE_DIR" >&2
  echo "Run: bash runpod/bootstrap_nnue_tools.sh" >&2
  exit 1
fi

mkdir -p "$DATASET_DIR" "$RUN_DIR" "$NET_DIR"

if [ ! -f "$BINPACK_FILE" ]; then
  echo "Converting $PLAIN_FILE to $BINPACK_FILE"
  "$STOCKFISH_BIN" convert "$PLAIN_FILE" "$BINPACK_FILE" validate
else
  echo "Using existing binpack: $BINPACK_FILE"
fi

cd "$NNUE_DIR"

if ! compgen -G "$NNUE_DIR/build/*training_data_loader.*" > /dev/null; then
  echo "Missing nnue-pytorch native data loader. Building it now..."
  if [ -x ./compile_data_loader.sh ]; then
    bash ./compile_data_loader.sh "$BINPACK_FILE"
  else
    echo "Missing compile_data_loader.sh in $NNUE_DIR" >&2
    exit 1
  fi
fi

if ! compgen -G "$NNUE_DIR/build/*training_data_loader.*" > /dev/null \
  && compgen -G "$NNUE_DIR/*training_data_loader.*" > /dev/null; then
  echo "Copying data loader shared library into build/ for _native.py"
  mkdir -p "$NNUE_DIR/build"
  cp "$NNUE_DIR"/*training_data_loader.* "$NNUE_DIR/build/"
fi

if ! compgen -G "$NNUE_DIR/build/*training_data_loader.*" > /dev/null; then
  echo "Still cannot find build/*training_data_loader.* after compile." >&2
  find "$NNUE_DIR" -name '*training_data_loader*' -type f -maxdepth 4 >&2 || true
  exit 1
fi

python train.py "$BINPACK_FILE" \
  --default-root-dir "$RUN_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --epoch-size "$EPOCH_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --validation-size 0 \
  --network-save-period "$NETWORK_SAVE_PERIOD" \
  --gpus "$GPUS" \
  --accelerator cuda

LATEST_CKPT="$(find "$RUN_DIR" -name '*.ckpt' -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [ -z "$LATEST_CKPT" ]; then
  echo "No checkpoint found under $RUN_DIR" >&2
  exit 1
fi

python serialize.py "$LATEST_CKPT" "$OUT_NNUE" \
  --description "goingawall1 eval baseline from Chess.com games"

echo "Wrote $OUT_NNUE"
