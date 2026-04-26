#!/usr/bin/env python3
"""Build preference pairs from current-engine move disagreements."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import chess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create hard chosen-vs-rejected pairs where rejected is the engine's move."
    )
    parser.add_argument("--moves", default="data/goingawall1/player_moves.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--movetime-ms", type=int, default=None)
    parser.add_argument("--min-ply", type=int, default=12)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=120)
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
            return line
    if proc.poll() is not None:
        raise RuntimeError(
            "Engine exited while waiting for {0}. Last lines: {1}".format(prefix, lines[-10:])
        )
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


def best_move(proc, fen, depth, movetime_ms, timeout_seconds):
    send(proc, "position fen {0}".format(fen))
    if movetime_ms is not None:
        send(proc, "go movetime {0}".format(movetime_ms))
    else:
        send(proc, "go depth {0}".format(depth))
    line = read_until(proc, "bestmove", timeout_seconds=timeout_seconds)
    parts = line.split()
    if len(parts) < 2:
        raise RuntimeError("Malformed bestmove line: {0}".format(line))
    return parts[1]


def color_name(color):
    return "white" if color == chess.WHITE else "black"


def build_pair(row, engine_move_uci):
    played_move_uci = row["played_move_uci"]
    if engine_move_uci == played_move_uci or engine_move_uci in ("none", "(none)"):
        return None

    board = chess.Board(row["fen_before"])
    try:
        played_move = chess.Move.from_uci(played_move_uci)
        engine_move = chess.Move.from_uci(engine_move_uci)
    except ValueError:
        return None
    if played_move not in board.legal_moves or engine_move not in board.legal_moves:
        return None

    chosen_board = board.copy(stack=False)
    chosen_board.push(played_move)
    rejected_board = board.copy(stack=False)
    rejected_board.push(engine_move)

    return {
        "position_id": row["position_id"],
        "game_id": row.get("game_id"),
        "url": row.get("url"),
        "ply": row["ply"],
        "fen_before": row["fen_before"],
        "chosen_move_uci": played_move_uci,
        "chosen_after_fen": chosen_board.fen(),
        "rejected_move_uci": engine_move_uci,
        "rejected_after_fen": rejected_board.fen(),
        "rejected_source": "engine_best",
        "target_username": row.get("target_username"),
        "target_color": row["target_color"],
        "side_to_move": row.get("side_to_move", color_name(board.turn)),
        "result_side_to_move": row.get("result_side_to_move"),
        "policy_alpha": row.get("policy_alpha"),
        "policy_weight": row.get("policy_weight"),
        "eval_weight": row.get("eval_weight"),
        "preference_margin_cp": 25,
    }


def main():
    args = parse_args()
    rows = load_rows(args.moves, args.limit, args.min_ply, args.time_class)
    if not rows:
        raise SystemExit("No rows selected.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [args.stockfish],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    checked = 0
    matches = 0
    written = 0
    skipped = 0
    try:
        send(proc, "uci")
        read_until(proc, "uciok")
        send(proc, "setoption name Threads value {0}".format(args.threads))
        send(proc, "setoption name Hash value {0}".format(args.hash))
        send(proc, "setoption name EvalFile value {0}".format(args.eval_file))
        send(proc, "isready")
        read_until(proc, "readyok")

        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for checked, row in enumerate(rows, start=1):
                engine_move = best_move(
                    proc,
                    row["fen_before"],
                    args.depth,
                    args.movetime_ms,
                    args.timeout_seconds,
                )
                if engine_move == row["played_move_uci"]:
                    matches += 1
                pair = build_pair(row, engine_move)
                if pair is None:
                    skipped += 1
                else:
                    handle.write(json.dumps(pair, sort_keys=True))
                    handle.write("\n")
                    written += 1
                if args.progress_every and checked % args.progress_every == 0:
                    print(
                        "checked={0} pairs={1} agreement={2:.2%}".format(
                            checked, written, matches / checked
                        ),
                        flush=True,
                    )
    finally:
        if proc.poll() is None:
            try:
                send(proc, "quit")
            except BrokenPipeError:
                pass
            proc.terminate()

    result = {
        "moves_file": args.moves,
        "out": str(out_path),
        "stockfish": args.stockfish,
        "eval_file": args.eval_file,
        "depth": args.depth,
        "movetime_ms": args.movetime_ms,
        "threads": args.threads,
        "hash": args.hash,
        "checked": checked,
        "matches": matches,
        "agreement": matches / checked if checked else 0.0,
        "pairs_written": written,
        "skipped": skipped,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
