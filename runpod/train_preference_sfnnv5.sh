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

BASE_NNUE="${BASE_NNUE:-$WORK_DIR/nets/goingawall1_current_best.nnue}"
if [ ! -f "$BASE_NNUE" ]; then
  BASE_NNUE="${FALLBACK_BASE_NNUE:-$WORK_DIR/nets/ft_lr1e-5_e8_l02.nnue}"
fi
if [ ! -f "$BASE_NNUE" ]; then
  BASE_NNUE="$DEFAULT_NET"
fi

RUN_NAME="${RUN_NAME:-pref_alpha08_lr1e-6_e2_s300}"
RUN_DIR="${RUN_DIR:-$WORK_DIR/runs/$RUN_NAME}"
PAIRS_FILE="${PAIRS_FILE:-$DATA_DIR/preference_pairs.jsonl}"
BASE_PT="${BASE_PT:-$RUN_DIR/base.pt}"
OUT_PT="${OUT_PT:-$RUN_DIR/$RUN_NAME.pt}"
OUT_NNUE="${OUT_NNUE:-$WORK_DIR/nets/$RUN_NAME.nnue}"

PREF_EPOCHS="${PREF_EPOCHS:-2}"
PREF_STEPS="${PREF_STEPS:-300}"
PREF_BATCH_SIZE="${PREF_BATCH_SIZE:-512}"
PREF_LR="${PREF_LR:-1e-6}"
ALPHA="${ALPHA:-0.8}"
MARGIN_CP="${MARGIN_CP:-25}"
TEMPERATURE_CP="${TEMPERATURE_CP:-100}"
DISTILL_SCALE_CP="${DISTILL_SCALE_CP:-600}"
MAX_PAIRS="${MAX_PAIRS:-}"
TRAIN_FEATURE_TRANSFORMER="${TRAIN_FEATURE_TRANSFORMER:-False}"

FEATURES="${FEATURES:-HalfKAv2_hm^}"
L1="${L1:-1024}"
L2="${L2:-15}"
L3="${L3:-32}"
FT_COMPRESSION="${FT_COMPRESSION:-none}"

mkdir -p "$WORK_DIR/nets" "$WORK_DIR/tmp" "$RUN_DIR"
export TMPDIR="${TMPDIR:-$WORK_DIR/tmp}"

if [ ! -f "$PAIRS_FILE" ]; then
  echo "Missing preference pairs: $PAIRS_FILE" >&2
  exit 1
fi

if [ ! -f "$DEFAULT_NET" ]; then
  echo "Downloading default compatible net: $DEFAULT_NET_NAME"
  curl -L "$DEFAULT_NET_URL" -o "$DEFAULT_NET"
fi

cd "$NNUE_DIR"

if [ "$BASE_NNUE" = "$DEFAULT_NET" ] && [ -f "$DEFAULT_PT" ]; then
  cp "$DEFAULT_PT" "$BASE_PT"
else
  echo "Converting base net to PyTorch model: $BASE_NNUE -> $BASE_PT"
  python serialize.py "$BASE_NNUE" "$BASE_PT" \
    --features "$FEATURES" \
    --l1 "$L1" \
    --l2 "$L2" \
    --l3 "$L3"
fi

TRAIN_FT_ARG=()
if [ "${TRAIN_FEATURE_TRANSFORMER,,}" = "true" ]; then
  TRAIN_FT_ARG=(--train-feature-transformer)
fi

MAX_PAIRS_ARG=()
if [ -n "$MAX_PAIRS" ]; then
  MAX_PAIRS_ARG=(--max-pairs "$MAX_PAIRS")
fi

echo "Preference fine-tuning from $BASE_PT"
python /workspace/chess_v4/scripts/train_preference_nnue.py \
  --pairs "$PAIRS_FILE" \
  --base-model "$BASE_PT" \
  --out-model "$OUT_PT" \
  --epochs "$PREF_EPOCHS" \
  --steps-per-epoch "$PREF_STEPS" \
  --batch-size "$PREF_BATCH_SIZE" \
  --alpha "$ALPHA" \
  --margin-cp "$MARGIN_CP" \
  --temperature-cp "$TEMPERATURE_CP" \
  --distill-scale-cp "$DISTILL_SCALE_CP" \
  --lr "$PREF_LR" \
  "${TRAIN_FT_ARG[@]}" \
  "${MAX_PAIRS_ARG[@]}"

echo "Serializing preference model to $OUT_NNUE"
python serialize.py "$OUT_PT" "$OUT_NNUE" \
  --features "$FEATURES" \
  --l1 "$L1" \
  --l2 "$L2" \
  --l3 "$L3" \
  --ft-compression "$FT_COMPRESSION" \
  --description "goingawall1 preference fine-tune alpha=$ALPHA base=$(basename "$BASE_NNUE")"

echo "Wrote $OUT_NNUE"

cd /workspace/chess_v4
bash runpod/test_nnue_smoke.sh "$OUT_NNUE"

python scripts/measure_move_agreement.py \
  --stockfish "$STOCKFISH_BIN" \
  --eval-file "$OUT_NNUE" \
  --moves "$DATA_DIR/player_moves.jsonl" \
  --limit 500 \
  --depth 6 \
  --min-ply 12 \
  --progress-every 50
