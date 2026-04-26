#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${RUNPOD_WORK_DIR:-/workspace/work}"
STOCKFISH_BIN="${STOCKFISH_BIN:-$WORK_DIR/Stockfish-tools/src/stockfish}"
NET_FILE="${1:-${OUT_NNUE:-$WORK_DIR/nets/goingawall1_eval.nnue}}"

if [ ! -x "$STOCKFISH_BIN" ]; then
  echo "Missing Stockfish binary: $STOCKFISH_BIN" >&2
  exit 1
fi

if [ ! -f "$NET_FILE" ]; then
  echo "Missing NNUE file: $NET_FILE" >&2
  exit 1
fi

"$STOCKFISH_BIN" <<EOF
uci
setoption name EvalFile value $NET_FILE
isready
position startpos moves e2e4 e7e5 g1f3 b8c6
go depth 8
quit
EOF
