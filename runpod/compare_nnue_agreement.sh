#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${RUNPOD_WORK_DIR:-/workspace/work}"
STOCKFISH_BIN="${STOCKFISH_BIN:-$WORK_DIR/Stockfish-tools/src/stockfish}"
MOVES_FILE="${MOVES_FILE:-data/goingawall1/player_moves.jsonl}"
LIMIT="${LIMIT:-500}"
DEPTH="${DEPTH:-6}"
MIN_PLY="${MIN_PLY:-12}"

if [ "$#" -eq 0 ]; then
  set -- \
    "default=nn-3c0aa92af1da.nnue" \
    "scratch=$WORK_DIR/nets/goingawall1_sfnnv5_fixed.nnue" \
    "latest=$WORK_DIR/nets/ft_best_e5.nnue"
fi

echo "name,eval_file,limit,depth,min_ply,agreement,matches,checked"

for item in "$@"; do
  name="${item%%=*}"
  eval_file="${item#*=}"
  tmp="$(mktemp)"
  python scripts/measure_move_agreement.py \
    --stockfish "$STOCKFISH_BIN" \
    --eval-file "$eval_file" \
    --moves "$MOVES_FILE" \
    --limit "$LIMIT" \
    --depth "$DEPTH" \
    --min-ply "$MIN_PLY" > "$tmp"

  python - "$tmp" "$name" "$eval_file" "$LIMIT" "$DEPTH" "$MIN_PLY" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
start = text.rfind("\n{")
if start == -1:
    start = text.find("{")
else:
    start += 1
result = json.loads(text[start:])
print(
    "{},{},{},{},{},{:.4f},{},{}".format(
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        result["agreement"],
        result["matches"],
        result["checked"],
    )
)
PY
  rm -f "$tmp"
done
