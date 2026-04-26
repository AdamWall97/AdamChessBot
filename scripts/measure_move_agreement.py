#!/usr/bin/env python3
"""Measure how often a Stockfish+NNUE setup chooses the user's played move."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Measure move agreement on player_moves.jsonl.")
    parser.add_argument("--moves", default="data/goingawall1/player_moves.jsonl")
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--min-ply", type=int, default=0)
    parser.add_argument("--time-class", default=None)
    return parser.parse_args()


def send(proc, command):
    proc.stdin.write(command + "\n")
    proc.stdin.flush()


def read_until(proc, prefix, timeout_seconds=60):
    deadline = time.time() + timeout_seconds
    lines = []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        lines.append(line)
        if line.startswith(prefix):
            return line, lines
    raise RuntimeError("Timed out waiting for {0}. Last lines: {1}".format(prefix, lines[-10:]))


def load_rows(path, limit, min_ply, time_class):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["ply"] < min_ply:
                continue
            if time_class and row.get("time_class") != time_class:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def best_move(proc, fen, depth):
    send(proc, "position fen {0}".format(fen))
    send(proc, "go depth {0}".format(depth))
    line, _ = read_until(proc, "bestmove", timeout_seconds=120)
    parts = line.split()
    if len(parts) < 2:
        raise RuntimeError("Malformed bestmove line: {0}".format(line))
    return parts[1]


def main():
    args = parse_args()
    rows = load_rows(args.moves, args.limit, args.min_ply, args.time_class)
    if not rows:
        raise SystemExit("No rows selected.")

    proc = subprocess.Popen(
        [args.stockfish],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        send(proc, "uci")
        read_until(proc, "uciok")
        send(proc, "setoption name EvalFile value {0}".format(args.eval_file))
        send(proc, "isready")
        read_until(proc, "readyok")

        matches = 0
        samples = []
        for index, row in enumerate(rows, start=1):
            engine_move = best_move(proc, row["fen_before"], args.depth)
            played_move = row["played_move_uci"]
            matched = engine_move == played_move
            matches += 1 if matched else 0
            if len(samples) < 20:
                samples.append(
                    {
                        "position_id": row["position_id"],
                        "ply": row["ply"],
                        "fen": row["fen_before"],
                        "played": played_move,
                        "engine": engine_move,
                        "matched": matched,
                        "url": row["url"],
                    }
                )
            if index % 50 == 0:
                print("checked={0} agreement={1:.2%}".format(index, matches / index), flush=True)

        result = {
            "moves_file": args.moves,
            "stockfish": args.stockfish,
            "eval_file": args.eval_file,
            "depth": args.depth,
            "checked": len(rows),
            "matches": matches,
            "agreement": matches / len(rows),
            "sample": samples,
        }
        print(json.dumps(result, indent=2))
    finally:
        if proc.poll() is None:
            try:
                send(proc, "quit")
            except BrokenPipeError:
                pass
            proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
