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

Use a Python 3.12 PyTorch image if possible. Current `official-stockfish/nnue-pytorch` uses Python 3.12 syntax, so older RunPod templates like `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` can fail unless patched.

Best container image:

```text
nvcr.io/nvidia/pytorch:25.03-py3
```

That is the base image used by the official `nnue-pytorch` NVIDIA Dockerfile. On RunPod, create a custom template with this container image, or use the closest available PyTorch template that has Python 3.12 and CUDA 12.x.

For custom templates, set the container start command to keep the pod alive:

```text
sleep infinity
```

or:

```text
/bin/bash -lc "sleep infinity"
```

Without a long-running start command, the container can exit immediately and SSH will fail with `container ... is not running`.

Create a GPU pod from that PyTorch/CUDA template, attach enough disk for the repo plus build artifacts, then run:

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

## Iterating And Testing

After each training run, first confirm Stockfish can load the net:

```bash
bash runpod/test_nnue_smoke.sh /workspace/work/nets/goingawall1_eval.nnue
```

Then measure move agreement against your actual Chess.com moves:

```bash
python scripts/measure_move_agreement.py \
  --stockfish /workspace/work/Stockfish-tools/src/stockfish \
  --eval-file /workspace/work/nets/goingawall1_eval.nnue \
  --moves data/goingawall1/player_moves.jsonl \
  --limit 500 \
  --depth 6
```

Good iteration loop:

1. Train a net.
2. Smoke-test that Stockfish can load it.
3. Measure move agreement.
4. Play a few short games against the net.
5. Change one training variable at a time.

The best next experiment is to fine-tune from the compatible default Stockfish net instead of training from scratch:

```bash
bash runpod/finetune_sfnnv5_from_default.sh
```

This downloads `nn-3c0aa92af1da.nnue`, converts it to a PyTorch model, fine-tunes it on `personal.binpack`, serializes a new `.nnue`, smoke-tests it, and measures agreement from move 7 onward.

Good first sweep:

```bash
RUN_NAME=ft_lr1e-5_e2 LR=1e-5 MAX_EPOCHS=2 bash runpod/finetune_sfnnv5_from_default.sh
RUN_NAME=ft_lr5e-5_e2 LR=5e-5 MAX_EPOCHS=2 bash runpod/finetune_sfnnv5_from_default.sh
RUN_NAME=ft_lr1e-4_e2 LR=1e-4 MAX_EPOCHS=2 bash runpod/finetune_sfnnv5_from_default.sh
```

Keep the run that improves agreement without producing obviously strange opening or tactical choices.

Useful knobs:

- `MAX_EPOCHS`: more training time.
- `EPOCH_SIZE`: more batches per epoch.
- `BATCH_SIZE`: larger if VRAM allows.
- `NETWORK_SAVE_PERIOD`: save checkpoints less often to reduce disk usage.
- `SAVE_LAST_NETWORK=False`: avoids an extra `last.ckpt` copy.
- `FEATURES`, `L1`, `L2`, `L3`: must match the Stockfish binary architecture. The current script defaults to the SFNNv5-compatible settings expected by the `tools` branch binary: `HalfKAv2_hm^`, `L1=1024`, `L2=15`, `L3=32`.
- `FT_COMPRESSION=none`: safest for the older `tools` branch Stockfish binary. Current `nnue-pytorch` defaults to LEB128 compression, which may not load in older engine builds.
- `--depth` in `measure_move_agreement.py`: higher is slower but closer to real engine choice.

Do not judge the net only by move agreement. A personal NNUE should also avoid collapsing into bad chess, so track both agreement and practical game quality.

If Stockfish rejects a serialized `.nnue`, inspect the checkpoint before retrying serialization:

```bash
python scripts/inspect_nnue_checkpoint.py /path/to/checkpoint.ckpt --expect-sfnnv5
```

For the `tools` branch binary with default `nn-3c0aa92af1da.nnue`, the checkpoint should have one input feature weight with shape `[24576, 1032]`, `l1.linear.weight` shape `[128, 1024]`, and `l2.linear.weight` shape `[256, 30]`.
