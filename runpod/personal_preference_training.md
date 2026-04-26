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

## Implemented Preference Trainer

The repo includes a first practical preference trainer:

```bash
bash runpod/train_preference_sfnnv5.sh
```

It:

1. Loads `preference_pairs.jsonl`.
2. Converts `chosen_after_fen` and `rejected_after_fen` with the nnue-pytorch native FEN loader.
3. Runs the same NNUE model on both after-positions.
4. Applies the pairwise preference loss.
5. Uses teacher distillation against the base net for the `0.2` chess-sanity term.
6. Saves `.pt`, serializes `.nnue`, smoke-tests it, and measures agreement.

The important compatibility constraint: keep the model architecture and feature set supported by `serialize.py`, otherwise the output may be a PyTorch checkpoint but not a Stockfish-loadable `.nnue`.
