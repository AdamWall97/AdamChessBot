#!/usr/bin/env python3
"""Prepare Chess.com games for personal NNUE training experiments."""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import chess
import chess.pgn


API_TEMPLATE = "https://api.chess.com/pub/player/{username}/games/{year:04d}/{month:02d}"
DEFAULT_USERNAME = "goingawall1"
DEFAULT_ALPHA = 0.8
DEFAULT_NEGATIVE_SAMPLES = 8
DEFAULT_RESULT_SCORE_CP = 400

PIECE_VALUES_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def parse_args():
    today = dt.date.today()
    default_start = today - dt.timedelta(days=730)

    parser = argparse.ArgumentParser(
        description="Fetch Chess.com games and build NNUE/move-imitation datasets."
    )
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--start-date", default=default_start.isoformat())
    parser.add_argument("--end-date", default=today.isoformat())
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--negative-samples", type=int, default=DEFAULT_NEGATIVE_SAMPLES)
    parser.add_argument("--preference-margin-cp", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument(
        "--score-mode",
        choices=("outcome", "zero", "material"),
        default="outcome",
        help="Pseudo score target for nnue_plain.plain before engine analysis exists.",
    )
    parser.add_argument("--result-score-cp", type=int, default=DEFAULT_RESULT_SCORE_CP)
    parser.add_argument("--refresh", action="store_true", help="Re-download cached months.")
    parser.add_argument("--sleep-seconds", type=float, default=0.35)
    parser.add_argument(
        "--include-opponent-moves",
        action="store_true",
        help="Also emit opponent moves. Default is only target user's moves.",
    )
    return parser.parse_args()


def date_from_arg(value):
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit("Expected date in YYYY-MM-DD format: {0}".format(value))


def month_range(start_date, end_date):
    year = start_date.year
    month = start_date.month
    while (year, month) <= (end_date.year, end_date.month):
        yield year, month
        month += 1
        if month == 13:
            year += 1
            month = 1


def utc_date_from_epoch(epoch_seconds):
    return dt.datetime.fromtimestamp(int(epoch_seconds), dt.timezone.utc).date()


def fetch_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "chess-v4-nnue-data-prep/0.1 (personal research)",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def load_archive(username, year, month, raw_dir, refresh):
    cache_path = raw_dir / "{0:04d}-{1:02d}.json".format(year, month)
    if cache_path.exists() and not refresh:
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), False

    url = API_TEMPLATE.format(username=username.lower(), year=year, month=month)
    try:
        payload = fetch_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            payload = {"games": []}
        else:
            raise

    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload, True


def normalize_name(value):
    return (value or "").strip().lower()


def chesscom_player_color(game_json, username, game):
    target = normalize_name(username)
    white_api = normalize_name(game_json.get("white", {}).get("username"))
    black_api = normalize_name(game_json.get("black", {}).get("username"))
    if white_api == target:
        return chess.WHITE
    if black_api == target:
        return chess.BLACK

    white_header = normalize_name(game.headers.get("White"))
    black_header = normalize_name(game.headers.get("Black"))
    if white_header == target:
        return chess.WHITE
    if black_header == target:
        return chess.BLACK
    return None


def result_for_color(result, color):
    if result == "1-0":
        return 1 if color == chess.WHITE else -1
    if result == "0-1":
        return -1 if color == chess.WHITE else 1
    if result == "1/2-1/2":
        return 0
    return None


def color_name(color):
    return "white" if color == chess.WHITE else "black"


def material_score_cp(board):
    white_score = 0
    black_score = 0
    for piece_type, value in PIECE_VALUES_CP.items():
        white_score += len(board.pieces(piece_type, chess.WHITE)) * value
        black_score += len(board.pieces(piece_type, chess.BLACK)) * value
    score = white_score - black_score
    return score if board.turn == chess.WHITE else -score


def pseudo_score_cp(board, side_result, score_mode, result_score_cp):
    if score_mode == "zero":
        return 0
    if score_mode == "material":
        return material_score_cp(board)
    return side_result * result_score_cp


def stable_id(*parts):
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:16]


def parse_pgn(game_json):
    pgn_text = game_json.get("pgn") or ""
    if not pgn_text.strip():
        return None
    return chess.pgn.read_game(io.StringIO(pgn_text))


def game_end_date(game_json):
    if "end_time" not in game_json:
        return None
    return utc_date_from_epoch(game_json["end_time"])


def player_api_info(game_json, color):
    key = color_name(color)
    return game_json.get(key, {})


def opponent_api_info(game_json, color):
    key = color_name(not color)
    return game_json.get(key, {})


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_records(game_json, game, username, args, rng):
    player_color = chesscom_player_color(game_json, username, game)
    if player_color is None:
        return [], [], "username_not_in_game"

    result_text = game.headers.get("Result", game_json.get("result", "*"))
    player_result = result_for_color(result_text, player_color)
    if player_result is None:
        return [], [], "unknown_result"

    rules = game_json.get("rules", "chess")
    if rules != "chess":
        return [], [], "non_standard_rules"

    board = game.board()
    records = []
    preference_pairs = []
    game_url = game_json.get("url") or game.headers.get("Link") or ""
    game_id = stable_id(game_url, game.headers.get("Date"), game.headers.get("Round"))
    end_date = game_end_date(game_json)
    player_info = player_api_info(game_json, player_color)
    opponent_info = opponent_api_info(game_json, player_color)
    opponent_color = not player_color

    for node in game.mainline():
        move = node.move
        if move not in board.legal_moves:
            return records, preference_pairs, "illegal_move_in_pgn"

        active_color = board.turn
        is_target_move = active_color == player_color
        if args.include_opponent_moves:
            should_emit = True
        else:
            should_emit = is_target_move

        san = board.san(move)

        if should_emit:
            side_result = player_result if active_color == player_color else -player_result
            fen_before = board.fen()
            legal_moves = sorted(legal_move.uci() for legal_move in board.legal_moves)
            after_board = board.copy(stack=False)
            after_board.push(move)
            fen_after = after_board.fen()
            record_id = stable_id(game_id, board.ply(), fen_before, move.uci())
            score_cp = pseudo_score_cp(board, side_result, args.score_mode, args.result_score_cp)
            alternatives = [candidate for candidate in board.legal_moves if candidate != move]
            sampled_alternatives = []
            if args.negative_samples > 0 and alternatives:
                count = min(args.negative_samples, len(alternatives))
                sampled_alternatives = rng.sample(list(alternatives), count)

            record = {
                "position_id": record_id,
                "game_id": game_id,
                "url": game_url,
                "end_date": end_date.isoformat() if end_date else None,
                "time_class": game_json.get("time_class"),
                "time_control": game_json.get("time_control"),
                "rated": game_json.get("rated"),
                "rules": rules,
                "ply": board.ply(),
                "move_number": board.fullmove_number,
                "side_to_move": color_name(active_color),
                "target_username": username,
                "target_color": color_name(player_color),
                "is_target_move": is_target_move,
                "opponent_username": opponent_info.get("username"),
                "target_rating": safe_int(player_info.get("rating")),
                "opponent_rating": safe_int(opponent_info.get("rating")),
                "game_result": result_text,
                "result_side_to_move": side_result,
                "result_target": player_result,
                "fen_before": fen_before,
                "played_move_uci": move.uci(),
                "played_move_san": san,
                "fen_after": fen_after,
                "legal_moves": legal_moves,
                "legal_move_count": len(legal_moves),
                "score_cp": score_cp,
                "score_mode": args.score_mode,
                "policy_alpha": args.alpha,
                "policy_weight": args.alpha,
                "eval_weight": round(1.0 - args.alpha, 10),
            }
            records.append(record)

            for rejected in sampled_alternatives:
                rejected_board = board.copy(stack=False)
                rejected_board.push(rejected)
                preference_pairs.append(
                    {
                        "position_id": record_id,
                        "game_id": game_id,
                        "url": game_url,
                        "ply": board.ply(),
                        "fen_before": fen_before,
                        "chosen_move_uci": move.uci(),
                        "chosen_after_fen": fen_after,
                        "rejected_move_uci": rejected.uci(),
                        "rejected_after_fen": rejected_board.fen(),
                        "target_username": username,
                        "target_color": color_name(player_color),
                        "side_to_move": color_name(active_color),
                        "result_side_to_move": side_result,
                        "policy_alpha": args.alpha,
                        "policy_weight": args.alpha,
                        "eval_weight": round(1.0 - args.alpha, 10),
                        "preference_margin_cp": args.preference_margin_cp,
                    }
                )

        board.push(move)

    return records, preference_pairs, None


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_csv(path, rows):
    fields = [
        "position_id",
        "game_id",
        "url",
        "end_date",
        "time_class",
        "time_control",
        "ply",
        "move_number",
        "side_to_move",
        "target_color",
        "opponent_username",
        "target_rating",
        "opponent_rating",
        "game_result",
        "result_side_to_move",
        "fen_before",
        "played_move_uci",
        "played_move_san",
        "fen_after",
        "legal_move_count",
        "score_cp",
        "policy_alpha",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_plain(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write("fen {0}\n".format(row["fen_before"]))
            handle.write("move {0}\n".format(row["played_move_uci"]))
            handle.write("score {0}\n".format(row["score_cp"]))
            handle.write("ply {0}\n".format(row["ply"]))
            handle.write("result {0}\n".format(row["result_side_to_move"]))
            handle.write("e\n")


def escape_epd_text(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def write_epd(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            epd_position = " ".join(row["fen_before"].split()[:4])
            comment = "san={0} result={1} alpha={2}".format(
                row["played_move_san"], row["result_side_to_move"], row["policy_alpha"]
            )
            handle.write(
                '{0} bm {1}; id "{2}"; c0 "{3}";\n'.format(
                    epd_position,
                    row["played_move_uci"],
                    escape_epd_text(row["position_id"]),
                    escape_epd_text(comment),
                )
            )


def write_combined_pgn(path, games):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pgn_text in games:
            handle.write(pgn_text.strip())
            handle.write("\n\n")


def main():
    args = parse_args()
    username = args.username.strip()
    start_date = date_from_arg(args.start_date)
    end_date = date_from_arg(args.end_date)
    if start_date > end_date:
        raise SystemExit("--start-date must be on or before --end-date")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / username.lower()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    months = list(month_range(start_date, end_date))
    all_games = []
    downloaded_months = []
    cached_months = []
    month_errors = []

    for index, (year, month) in enumerate(months, start=1):
        try:
            payload, downloaded = load_archive(username, year, month, raw_dir, args.refresh)
            if downloaded:
                downloaded_months.append("{0:04d}-{1:02d}".format(year, month))
            else:
                cached_months.append("{0:04d}-{1:02d}".format(year, month))
            for game_json in payload.get("games", []):
                end_date_utc = game_end_date(game_json)
                if end_date_utc is None:
                    continue
                if start_date <= end_date_utc <= end_date:
                    all_games.append(game_json)
        except Exception as exc:
            month_errors.append(
                {
                    "month": "{0:04d}-{1:02d}".format(year, month),
                    "error": "{0}: {1}".format(exc.__class__.__name__, exc),
                }
            )
        if index < len(months) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    move_records = []
    preference_pairs = []
    combined_pgns = []
    skipped = {}

    for game_json in all_games:
        game = parse_pgn(game_json)
        if game is None:
            skipped["missing_pgn"] = skipped.get("missing_pgn", 0) + 1
            continue
        records, pairs, skip_reason = build_records(game_json, game, username, args, rng)
        if skip_reason:
            skipped[skip_reason] = skipped.get(skip_reason, 0) + 1
            if not records:
                continue
        move_records.extend(records)
        preference_pairs.extend(pairs)
        if game_json.get("pgn"):
            combined_pgns.append(game_json["pgn"])

    outputs = {
        "combined_pgn": str(out_dir / "combined.pgn"),
        "player_moves_jsonl": str(out_dir / "player_moves.jsonl"),
        "player_moves_csv": str(out_dir / "player_moves.csv"),
        "preference_pairs_jsonl": str(out_dir / "preference_pairs.jsonl"),
        "nnue_plain": str(out_dir / "nnue_plain.plain"),
        "epd": str(out_dir / "player_moves.epd"),
        "manifest": str(out_dir / "manifest.json"),
    }

    write_combined_pgn(Path(outputs["combined_pgn"]), combined_pgns)
    write_jsonl(Path(outputs["player_moves_jsonl"]), move_records)
    write_csv(Path(outputs["player_moves_csv"]), move_records)
    write_jsonl(Path(outputs["preference_pairs_jsonl"]), preference_pairs)
    write_plain(Path(outputs["nnue_plain"]), move_records)
    write_epd(Path(outputs["epd"]), move_records)

    manifest = {
        "username": username,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "date_range": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "inclusive": True,
        },
        "api_url_template": API_TEMPLATE,
        "months_requested": ["{0:04d}-{1:02d}".format(year, month) for year, month in months],
        "months_downloaded": downloaded_months,
        "months_cached": cached_months,
        "month_errors": month_errors,
        "games_in_range": len(all_games),
        "games_with_pgn_written": len(combined_pgns),
        "move_records": len(move_records),
        "preference_pairs": len(preference_pairs),
        "skipped_games": skipped,
        "settings": {
            "alpha": args.alpha,
            "policy_weight": args.alpha,
            "eval_weight": round(1.0 - args.alpha, 10),
            "negative_samples": args.negative_samples,
            "preference_margin_cp": args.preference_margin_cp,
            "score_mode": args.score_mode,
            "result_score_cp": args.result_score_cp,
            "include_opponent_moves": args.include_opponent_moves,
            "seed": args.seed,
        },
        "outputs": outputs,
        "notes": [
            "Only target-user moves are emitted unless --include-opponent-moves is set.",
            "nnue_plain.plain is Stockfish-style text data; convert to binpack later with Stockfish tooling.",
            "preference_pairs.jsonl is intended for imitation-heavy custom NNUE training.",
        ],
    }
    with Path(outputs["manifest"]).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 1 if month_errors else 0


if __name__ == "__main__":
    sys.exit(main())
