#!/usr/bin/env bash
set -euo pipefail

USERNAME="${USERNAME:-goingawall1}"
START_DATE="${START_DATE:-2024-04-26}"
END_DATE="${END_DATE:-2026-04-26}"
ALPHA="${ALPHA:-0.8}"
NEGATIVE_SAMPLES="${NEGATIVE_SAMPLES:-8}"
OUT_DIR="${OUT_DIR:-data/$USERNAME}"

python -m pip install -r requirements.txt

python scripts/prepare_chesscom_nnue_data.py \
  --username "$USERNAME" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  --alpha "$ALPHA" \
  --negative-samples "$NEGATIVE_SAMPLES" \
  --out-dir "$OUT_DIR"

echo "Prepared data in $OUT_DIR"
