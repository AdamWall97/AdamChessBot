# Chess.com to NNUE Data Prep

This repo prepares `goingawall1` Chess.com games for a later NNUE training run.

The main script downloads monthly public Chess.com archives, filters to the last two years, keeps only positions where the target user is the side to move, and writes:

- `combined.pgn`: all matching games.
- `player_moves.jsonl`: one record per move you played, including FEN before/after and legal moves.
- `preference_pairs.jsonl`: chosen-move-vs-legal-alternative pairs for imitation-heavy NNUE training.
- `nnue_plain.plain`: Stockfish-style text training records that can later be converted to binpack.
- `player_moves.csv`: a spreadsheet-friendly summary.
- `manifest.json`: counts, settings, and output paths.

## Quick Start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts\prepare_chesscom_nnue_data.py --username goingawall1 --alpha 0.8
```

By default the script uses the local date and exports the last 730 days. On April 26, 2026, that means games ending from April 26, 2024 through April 26, 2026.

## Why There Are Two Training Formats

Stockfish NNUE training data is evaluation-oriented: each record has a position, a move, a score, a ply, and a game result. The `move` helps preserve the played game continuation and compression flow, but the standard trainer does not learn a policy head that directly says "play this move."

To make an engine prefer your actual moves, the useful data is `preference_pairs.jsonl`: for each position, your move is the chosen action and sampled legal alternatives are rejected actions. A later custom NNUE training step can use an alpha like `0.8` to weight this imitation/preference loss more strongly than the usual engine-score loss.

The `nnue_plain.plain` file is still emitted because it is the bridge into Stockfish-style tooling. Its default score target is outcome-based, not a deep Stockfish evaluation. For a stronger final net, the next step should annotate these positions with Stockfish scores for the remaining `0.2` "right play" component.

## Stockfish Plain Format

Each record looks like:

```text
fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
move e2e4
score 400
ply 0
result 1
e
```

`score` and `result` are from the side-to-move perspective. Because this pipeline keeps only your moves, that is also your perspective.

## Useful Options

```powershell
.\.venv\Scripts\python scripts\prepare_chesscom_nnue_data.py `
  --username goingawall1 `
  --start-date 2024-04-26 `
  --end-date 2026-04-26 `
  --alpha 0.8 `
  --negative-samples 8 `
  --score-mode outcome
```

`--score-mode outcome` maps wins/draws/losses to `+400/0/-400` centipawns. Use `--score-mode zero` if you want the plain file to carry only game result signal until engine analysis is added.

## RunPod Training

The RunPod scaffold lives in `runpod/`.

Start with:

```bash
bash runpod/bootstrap_nnue_tools.sh
bash runpod/train_stockfish_eval_nnue.sh
```

See `runpod/README.md` for the full baseline `.nnue` flow and the follow-up preference-loss design needed for true `alpha=0.8` move imitation.
