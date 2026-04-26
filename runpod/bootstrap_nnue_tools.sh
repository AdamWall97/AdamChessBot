#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${RUNPOD_WORK_DIR:-/workspace/work}"
NNUE_REPO="${NNUE_REPO:-https://github.com/official-stockfish/nnue-pytorch.git}"
STOCKFISH_REPO="${STOCKFISH_REPO:-https://github.com/official-stockfish/Stockfish.git}"
NNUE_DIR="${NNUE_DIR:-$WORK_DIR/nnue-pytorch}"
STOCKFISH_DIR="${STOCKFISH_DIR:-$WORK_DIR/Stockfish-tools}"

mkdir -p "$WORK_DIR"

if [ ! -d "$NNUE_DIR/.git" ]; then
  git clone --depth 1 "$NNUE_REPO" "$NNUE_DIR"
fi

cd "$NNUE_DIR"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
from pathlib import Path

path = Path("data_loader/_native.py")
text = path.read_text()
text = text.replace(
    "type SparseBatchPtr = ctypes._Pointer[SparseBatch]",
    "SparseBatchPtr = ctypes.POINTER(SparseBatch)",
)
text = text.replace(
    "type FenBatchPtr = ctypes._Pointer[FenBatch]",
    "FenBatchPtr = ctypes.POINTER(FenBatch)",
)
text = text.replace(
    "SparseBatchPtr = ctypes._Pointer[SparseBatch]",
    "SparseBatchPtr = ctypes.POINTER(SparseBatch)",
)
text = text.replace(
    "FenBatchPtr = ctypes._Pointer[FenBatch]",
    "FenBatchPtr = ctypes.POINTER(FenBatch)",
)
path.write_text(text)
PY

if [ -x ./setup_script.sh ]; then
  bash ./setup_script.sh
else
  bash ./compile_data_loader.sh
fi

if [ ! -d "$STOCKFISH_DIR/.git" ]; then
  git clone --depth 1 --branch tools "$STOCKFISH_REPO" "$STOCKFISH_DIR" \
    || git clone --depth 1 "$STOCKFISH_REPO" "$STOCKFISH_DIR"
fi

cd "$STOCKFISH_DIR/src"
make -j"$(nproc)" build ARCH=x86-64-modern

echo "NNUE tools are ready:"
echo "  nnue-pytorch: $NNUE_DIR"
echo "  stockfish:    $STOCKFISH_DIR/src/stockfish"
