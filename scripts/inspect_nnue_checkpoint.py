#!/usr/bin/env python3
"""Inspect an nnue-pytorch Lightning checkpoint's architecture-ish tensor shapes."""

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect nnue-pytorch checkpoint shapes.")
    parser.add_argument("checkpoint")
    parser.add_argument("--expect-sfnnv5", action="store_true")
    return parser.parse_args()


def shape_of(state_dict, key):
    value = state_dict.get(key)
    if value is None:
        return None
    return list(value.shape)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    feature_weights = {
        key: list(value.shape)
        for key, value in state_dict.items()
        if key.startswith("model.input.features.") and key.endswith(".weight")
    }

    report = {
        "checkpoint": str(checkpoint_path),
        "feature_weights": feature_weights,
        "l1_linear_weight": shape_of(state_dict, "model.layer_stacks.l1.linear.weight"),
        "l1_linear_bias": shape_of(state_dict, "model.layer_stacks.l1.linear.bias"),
        "l2_linear_weight": shape_of(state_dict, "model.layer_stacks.l2.linear.weight"),
        "output_linear_weight": shape_of(state_dict, "model.layer_stacks.output.linear.weight"),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.expect_sfnnv5:
        expected = {
            "model.input.features.0.weight": [24576, 1032],
            "model.layer_stacks.l1.linear.weight": [128, 1024],
            "model.layer_stacks.l2.linear.weight": [256, 30],
        }
        errors = []
        if sorted(feature_weights) != ["model.input.features.0.weight"]:
            errors.append("expected only model.input.features.0.weight")
        for key, expected_shape in expected.items():
            actual = shape_of(state_dict, key)
            if actual != expected_shape:
                errors.append("{0}: expected {1}, got {2}".format(key, expected_shape, actual))
        if errors:
            for error in errors:
                print("ERROR: {0}".format(error), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
