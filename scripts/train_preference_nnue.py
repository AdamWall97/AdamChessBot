#!/usr/bin/env python3
"""Fine-tune an nnue-pytorch model with chosen-vs-rejected move preferences."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import data_loader.stream as stream


def parse_args():
    parser = argparse.ArgumentParser(description="Preference fine-tune an NNUE model.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--base-model", required=True, help="Input .pt model from nnue-pytorch serialize.py")
    parser.add_argument("--out-model", required=True, help="Output .pt model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--margin-cp", type=float, default=25.0)
    parser.add_argument("--temperature-cp", type=float, default=100.0)
    parser.add_argument("--distill-scale-cp", type=float, default=600.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--train-feature-transformer",
        action="store_true",
        help="Also update the sparse input feature transformer. Default trains dense layers only.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def read_pairs(path, max_pairs=None):
    pairs = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            chosen = row["chosen_after_fen"]
            rejected = row["rejected_after_fen"]
            target_color = row["target_color"]
            fen_turn = chosen.split()[1]
            sign = 1.0 if (fen_turn == "w") == (target_color == "white") else -1.0
            pairs.append((chosen, rejected, sign))
            if max_pairs and len(pairs) >= max_pairs:
                break
    if not pairs:
        raise SystemExit("No preference pairs loaded from {0}".format(path))
    return pairs


def sparse_batch_from_fens(feature_set, fens, device):
    n = len(fens)
    ptr = stream.get_sparse_batch_from_fens(
        feature_set,
        fens,
        [0] * n,
        [0] * n,
        [0] * n,
    )
    try:
        return ptr.contents.get_tensors(device)
    finally:
        stream.destroy_sparse_batch(ptr)


def eval_cp(nnue, feature_set, fens, device):
    batch = sparse_batch_from_fens(feature_set, fens, device)
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        _outcome,
        _score,
        psqt_indices,
        layer_stack_indices,
    ) = batch
    raw = nnue.model(
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        psqt_indices,
        layer_stack_indices,
    )
    return (raw * nnue.model.quantization.nnue2score).squeeze(-1)


def set_trainable(nnue, train_feature_transformer):
    for param in nnue.parameters():
        param.requires_grad = True
    if not train_feature_transformer:
        for param in nnue.model.input.parameters():
            param.requires_grad = False


def trainable_parameters(nnue):
    return [param for param in nnue.parameters() if param.requires_grad]


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading preference pairs from {0}".format(args.pairs), flush=True)
    pairs = read_pairs(args.pairs, args.max_pairs)
    print("Loaded {0} preference pairs".format(len(pairs)), flush=True)

    print("Loading base model {0}".format(args.base_model), flush=True)
    nnue = torch.load(args.base_model, weights_only=False, map_location="cpu")
    teacher = torch.load(args.base_model, weights_only=False, map_location="cpu")
    nnue.to(device)
    teacher.to(device)
    nnue.train()
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    set_trainable(nnue, args.train_feature_transformer)
    optimizer = torch.optim.AdamW(
        trainable_parameters(nnue),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    feature_set = nnue.model.input_feature_name
    eval_weight = 1.0 - args.alpha
    print(
        "Preference training: alpha={0}, eval_weight={1}, margin_cp={2}, lr={3}, feature_set={4}, train_ft={5}".format(
            args.alpha,
            eval_weight,
            args.margin_cp,
            args.lr,
            feature_set,
            args.train_feature_transformer,
        ),
        flush=True,
    )

    for epoch in range(args.epochs):
        epoch_pref = 0.0
        epoch_distill = 0.0
        epoch_total = 0.0
        epoch_match_margin = 0.0
        for step in range(1, args.steps_per_epoch + 1):
            batch_rows = random.choices(pairs, k=args.batch_size)
            chosen_fens = [row[0] for row in batch_rows]
            rejected_fens = [row[1] for row in batch_rows]
            signs = torch.tensor(
                [row[2] for row in batch_rows],
                dtype=torch.float32,
                device=device,
            )

            chosen_cp = eval_cp(nnue, feature_set, chosen_fens, device)
            rejected_cp = eval_cp(nnue, feature_set, rejected_fens, device)
            with torch.no_grad():
                teacher_chosen_cp = eval_cp(teacher, feature_set, chosen_fens, device)
                teacher_rejected_cp = eval_cp(teacher, feature_set, rejected_fens, device)

            chosen_player_cp = chosen_cp * signs
            rejected_player_cp = rejected_cp * signs
            preference_margin = chosen_player_cp - rejected_player_cp
            preference_loss = F.softplus(
                (args.margin_cp - preference_margin) / args.temperature_cp
            ).mean()

            distill_loss = 0.5 * (
                torch.square((chosen_cp - teacher_chosen_cp) / args.distill_scale_cp).mean()
                + torch.square((rejected_cp - teacher_rejected_cp) / args.distill_scale_cp).mean()
            )
            loss = args.alpha * preference_loss + eval_weight * distill_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(nnue), 1.0)
            optimizer.step()
            if hasattr(nnue.model, "clip_weights"):
                with torch.no_grad():
                    nnue.model.clip_weights()

            epoch_pref += preference_loss.item()
            epoch_distill += distill_loss.item()
            epoch_total += loss.item()
            epoch_match_margin += preference_margin.mean().item()

            if args.log_every and step % args.log_every == 0:
                denom = float(step)
                print(
                    "epoch={0} step={1}/{2} total={3:.5f} pref={4:.5f} distill={5:.5f} avg_margin_cp={6:.2f}".format(
                        epoch + 1,
                        step,
                        args.steps_per_epoch,
                        epoch_total / denom,
                        epoch_pref / denom,
                        epoch_distill / denom,
                        epoch_match_margin / denom,
                    ),
                    flush=True,
                )

        denom = float(args.steps_per_epoch)
        print(
            "epoch={0} done total={1:.5f} pref={2:.5f} distill={3:.5f} avg_margin_cp={4:.2f}".format(
                epoch + 1,
                epoch_total / denom,
                epoch_pref / denom,
                epoch_distill / denom,
                epoch_match_margin / denom,
            ),
            flush=True,
        )

    nnue.cpu()
    nnue.eval()
    out = Path(args.out_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(nnue, out)
    print("Wrote {0}".format(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
