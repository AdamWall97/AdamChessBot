#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${RUNPOD_WORK_DIR:-/workspace/work}"
DATA_DIR="${RUNPOD_DATA_DIR:-/workspace/chess_v4/data/goingawall1}"
NNUE_DIR="${NNUE_DIR:-$WORK_DIR/nnue-pytorch}"
STOCKFISH_BIN="${STOCKFISH_BIN:-$WORK_DIR/Stockfish-tools/src/stockfish}"

DEFAULT_NET_NAME="${DEFAULT_NET_NAME:-nn-3c0aa92af1da.nnue}"
DEFAULT_NET_URL="${DEFAULT_NET_URL:-https://tests.stockfishchess.org/api/nn/$DEFAULT_NET_NAME}"
DEFAULT_NET="$WORK_DIR/nets/$DEFAULT_NET_NAME"
DEFAULT_PT="$WORK_DIR/nets/${DEFAULT_NET_NAME%.nnue}.pt"

DATASET="${DATASET:-$WORK_DIR/datasets/personal.binpack}"
RUN_NAME="${RUN_NAME:-goingawall1_ft_sfnnv5_lr5e-5_e2}"
RUN_DIR="${RUN_DIR:-$WORK_DIR/runs/$RUN_NAME}"
OUT_NNUE="${OUT_NNUE:-$WORK_DIR/nets/$RUN_NAME.nnue}"

MAX_EPOCHS="${MAX_EPOCHS:-2}"
EPOCH_SIZE="${EPOCH_SIZE:-50000}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NETWORK_SAVE_PERIOD="${NETWORK_SAVE_PERIOD:-1}"
SAVE_LAST_NETWORK="${SAVE_LAST_NETWORK:-False}"
LR="${LR:-5e-5}"
LAMBDA="${LAMBDA:-0.2}"
GPUS="${GPUS:-0}"

FEATURES="${FEATURES:-HalfKAv2_hm^}"
L1="${L1:-1024}"
L2="${L2:-15}"
L3="${L3:-32}"
FT_COMPRESSION="${FT_COMPRESSION:-none}"

mkdir -p "$WORK_DIR/nets" "$WORK_DIR/tmp"
export TMPDIR="${TMPDIR:-$WORK_DIR/tmp}"

if [ ! -f "$DATASET" ]; then
  echo "Missing dataset: $DATASET" >&2
  echo "Run runpod/train_stockfish_eval_nnue.sh once to create it, or rerun data conversion." >&2
  exit 1
fi

if [ ! -f "$DEFAULT_NET" ]; then
  echo "Downloading default compatible net: $DEFAULT_NET_NAME"
  curl -L "$DEFAULT_NET_URL" -o "$DEFAULT_NET"
fi

cd "$NNUE_DIR"

if [ ! -f "$DEFAULT_PT" ]; then
  echo "Converting default net to PyTorch model: $DEFAULT_PT"
  python serialize.py "$DEFAULT_NET" "$DEFAULT_PT" \
    --features "$FEATURES" \
    --l1 "$L1" \
    --l2 "$L2" \
    --l3 "$L3"
fi

echo "Fine-tuning from $DEFAULT_PT"
python train.py "$DATASET" \
  --resume-from-model "$DEFAULT_PT" \
  --default-root-dir "$RUN_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --epoch-size "$EPOCH_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --validation-size 0 \
  --network-save-period "$NETWORK_SAVE_PERIOD" \
  --save-last-network "$SAVE_LAST_NETWORK" \
  --features "$FEATURES" \
  --l1 "$L1" \
  --l2 "$L2" \
  --l3 "$L3" \
  --lambda "$LAMBDA" \
  --lr "$LR" \
  --gpus "$GPUS" \
  --accelerator cuda

LATEST_CKPT="$(find "$RUN_DIR" -name '*.ckpt' -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [ -z "$LATEST_CKPT" ]; then
  echo "No checkpoint found under $RUN_DIR" >&2
  exit 1
fi

echo "Serializing $LATEST_CKPT to $OUT_NNUE"
python serialize.py "$LATEST_CKPT" "$OUT_NNUE" \
  --features "$FEATURES" \
  --l1 "$L1" \
  --l2 "$L2" \
  --l3 "$L3" \
  --ft-compression "$FT_COMPRESSION" \
  --description "goingawall1 fine-tuned from $DEFAULT_NET_NAME"

echo "Wrote $OUT_NNUE"

cd /workspace/chess_v4
bash runpod/test_nnue_smoke.sh "$OUT_NNUE"

python scripts/measure_move_agreement.py \
  --stockfish "$STOCKFISH_BIN" \
  --eval-file "$OUT_NNUE" \
  --moves "$DATA_DIR/player_moves.jsonl" \
  --limit 500 \
  --depth 6 \
  --min-ply 12
