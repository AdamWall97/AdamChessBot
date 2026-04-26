# RunPod NNUE Training

This folder is the RunPod-facing side of the project.

There are two related training goals:

1. Produce a real Stockfish-loadable `.nnue` file.
2. Bias that evaluator toward your moves using the prepared preference pairs and `alpha=0.8`.

Those are not exactly the same thing. Stockfish NNUE is a scalar position evaluator, not a move-policy network. The engine still chooses moves by search. So a "plays like me" net should learn:

```text
score(position after my move) > score(position after legal alternative)
```

while still keeping some conventional chess-strength signal:

```text
loss = 0.8 * preference_loss + 0.2 * evaluation_or_result_loss
```

## Practical Training Plan

### Stage 1: Baseline `.nnue`

Use official Stockfish tooling:

1. Convert `data/goingawall1/nnue_plain.plain` to `personal.binpack`.
2. Train with `official-stockfish/nnue-pytorch`.
3. Serialize the best/latest checkpoint to `goingawall1.nnue`.

This produces a true `.nnue`, but it is not yet the strongest "play my exact moves" version because official training is evaluation-oriented.

### Stage 2: Personal Preference Fine-Tune

Patch or extend `nnue-pytorch` with a pairwise loss over `preference_pairs.jsonl`:

```text
preference_loss = max(0, margin - eval(chosen_after_fen) + eval(rejected_after_fen))
eval_loss       = standard NNUE WDL / centipawn / game-result loss
total_loss      = 0.8 * preference_loss + 0.2 * eval_loss
```

This still trains a scalar evaluator, which means it can remain compatible with Stockfish search. The trick is that your chosen moves create after-positions the evaluator learns to prefer.

## RunPod Setup

Important: RunPod cannot access files on your local computer unless you put them somewhere the pod can reach.

You have three good options:

1. Push this repo to GitHub, clone it on the pod, then re-download the Chess.com data on the pod.
2. Upload the local `data/goingawall1/` folder to a RunPod network volume.
3. Build and push the Docker image to a registry, then use that image as a RunPod template.

For this project, option 1 is simplest because the Chess.com API is public and the pod can recreate the data.

```bash
git clone <your-repo-url> /workspace/chess_v4
cd /workspace/chess_v4
bash runpod/prepare_data_on_pod.sh
bash runpod/bootstrap_nnue_tools.sh
bash runpod/train_stockfish_eval_nnue.sh
```

## Cheapest GPU Recommendation

Use a Community Cloud 16GB GPU first, preferably `RTX A4000`.

Why:

- RunPod lists the 16GB `A4000, A4500, RTX 4000, RTX 2000` group as its most cost-effective small-model tier.
- The RTX A4000 page advertises pricing from about `$0.25/hr`.
- This NNUE job is small compared with LLM training: about 36k positions and 275k preference pairs. VRAM should not be the bottleneck.
- If A4000 availability is poor, the next practical cheap choices are `RTX 3090 24GB`, then `RTX 4090 24GB`.

Start with:

```bash
MAX_EPOCHS=20 EPOCH_SIZE=100000 BATCH_SIZE=4096 bash runpod/train_stockfish_eval_nnue.sh
```

If it fits and runs cleanly, increase to:

```bash
MAX_EPOCHS=40 EPOCH_SIZE=200000 BATCH_SIZE=8192 bash runpod/train_stockfish_eval_nnue.sh
```

Create a GPU pod from a PyTorch/CUDA template, attach enough disk for the repo plus build artifacts, then run:

```bash
git clone <your-repo-url> /workspace/chess_v4
cd /workspace/chess_v4
bash runpod/prepare_data_on_pod.sh
bash runpod/bootstrap_nnue_tools.sh
bash runpod/train_stockfish_eval_nnue.sh
```

If you build the Docker image yourself:

```bash
docker build -t chess-v4-nnue -f runpod/Dockerfile .
docker run --gpus all --ipc=host -it -v "$PWD":/workspace/chess_v4 chess-v4-nnue
```

Inside the container:

```bash
bash runpod/bootstrap_nnue_tools.sh
bash runpod/train_stockfish_eval_nnue.sh
```

## Expected Inputs

The training scripts expect:

```text
data/goingawall1/nnue_plain.plain
data/goingawall1/preference_pairs.jsonl
data/goingawall1/player_moves.jsonl
```

The `data/` folder is ignored by git, so upload/copy it to the pod separately if it is not already present.

## Outputs

Default output location:

```text
/workspace/work/
  nnue-pytorch/
  Stockfish-tools/
  datasets/
    personal.binpack
  runs/
    goingawall1_eval/
  nets/
    goingawall1_eval.nnue
```
