# Personal Preference Fine-Tuning Design

The baseline script trains a valid Stockfish `.nnue`, but it does not directly optimize "choose my move." To make the net personal, extend `nnue-pytorch` with an additional dataset and loss.

## Dataset

Use:

```text
data/goingawall1/preference_pairs.jsonl
```

Each row contains:

- `chosen_after_fen`: position after the move you actually played.
- `rejected_after_fen`: position after a sampled legal alternative.
- `policy_weight`: default `0.8`.
- `eval_weight`: default `0.2`.
- `preference_margin_cp`: default `25`.

## Loss

The NNUE still outputs a scalar evaluation. For a side-to-move perspective, train:

```text
preference_loss = relu(margin - eval(chosen_after_fen) + eval(rejected_after_fen))
eval_loss       = normal nnue-pytorch WDL/result loss on chosen_after_fen or fen_before
total_loss      = policy_weight * preference_loss + eval_weight * eval_loss
```

With `alpha=0.8`, a batch cares much more about matching your move than matching Stockfish's preferred move.

## Why This Works With Stockfish Search

Stockfish search compares move lines by evaluating resulting positions. If the evaluator consistently gives your chosen after-positions a higher value than alternatives in similar positions, search becomes more likely to choose moves in your style.

## Recommended Next Patch

Add a custom Lightning module or training loop that:

1. Loads `preference_pairs.jsonl`.
2. Converts both FENs in each pair into the same feature indices used by the selected NNUE feature set.
3. Runs the same NNUE model twice.
4. Applies the pairwise margin loss.
5. Optionally mixes in `personal.binpack` batches for the `0.2` evaluation term.
6. Serializes with `serialize.py` once the checkpoint is trained.

The important compatibility constraint: keep the model architecture and feature set supported by `serialize.py`, otherwise the output may be a PyTorch checkpoint but not a Stockfish-loadable `.nnue`.
