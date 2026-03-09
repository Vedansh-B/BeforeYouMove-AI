"""Generate labeled chess positions (fen, eval) using Stockfish."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from typing import List, Optional, Tuple

import chess
import chess.engine
import chess.pgn


logger = logging.getLogger(__name__)


def score_to_pawns(score: chess.engine.PovScore, mate_clamp: float = 10.0) -> float:
    """
    Convert engine score to scalar pawns from White's perspective.

    - Mate scores are clamped to +/- mate_clamp.
    - Centipawn scores are converted to pawns (cp / 100.0).
    """
    white_score = score.white()
    mate = white_score.mate()
    if mate is not None:
        if mate > 0:
            return mate_clamp
        if mate < 0:
            return -mate_clamp
        return 0.0

    cp = white_score.score()
    if cp is None:
        return 0.0
    return float(cp) / 100.0


def evaluate_position(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    depth: int,
    mate_clamp: float = 10.0,
) -> float:
    """Analyze one position with Stockfish and return eval in pawns (White perspective)."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"]
    return score_to_pawns(score, mate_clamp=mate_clamp)


def generate_random_positions(num_positions: int) -> List[chess.Board]:
    """
    Generate positions by random legal play.

    Starts from the initial position, plays a random number of plies,
    and samples intermediate non-terminal positions.
    """
    positions: List[chess.Board] = []

    while len(positions) < num_positions:
        board = chess.Board()
        target_plies = random.randint(8, 80)

        for ply in range(target_plies):
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(random.choice(legal_moves))

            # Start collecting after a few plies to avoid too many near-initial duplicates.
            if ply >= 4 and not board.is_game_over():
                positions.append(board.copy(stack=False))
                if len(positions) >= num_positions:
                    break

    return positions


def generate_positions_from_pgn(pgn_path: str, num_positions: int) -> List[chess.Board]:
    """Sample positions from PGN games."""
    positions: List[chess.Board] = []

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while len(positions) < num_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.is_game_over():
                    continue
                positions.append(board.copy(stack=False))
                if len(positions) >= num_positions:
                    break

    if not positions:
        return positions

    # Random sample if we collected more than requested.
    if len(positions) > num_positions:
        positions = random.sample(positions, num_positions)
    return positions


def label_positions(
    positions: List[chess.Board],
    engine: chess.engine.SimpleEngine,
    depth: int,
    progress_every: int,
) -> List[Tuple[str, float]]:
    """Evaluate positions and return list of (fen, eval)."""
    rows: List[Tuple[str, float]] = []
    total = len(positions)

    for idx, board in enumerate(positions, start=1):
        fen = board.fen()
        eval_pawns = evaluate_position(board, engine, depth=depth)
        rows.append((fen, eval_pawns))

        if idx % progress_every == 0 or idx == total:
            logger.info("Labeled %d/%d positions", idx, total)

    return rows


def write_csv(rows: List[Tuple[str, float]], output_csv: str) -> None:
    """Write labeled dataset to CSV with header: fen,eval."""
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "eval"])
        for fen, eval_value in rows:
            writer.writerow([fen, f"{eval_value:.4f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chess training labels using Stockfish.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--num_positions", type=int, required=True, help="Number of positions to generate.")
    parser.add_argument("--stockfish_path", type=str, required=True, help="Path to local Stockfish binary.")
    parser.add_argument("--mode", type=str, choices=["random", "pgn"], required=True, help="Position source mode.")
    parser.add_argument("--pgn_path", type=str, default=None, help="PGN file path (required in pgn mode).")
    parser.add_argument("--depth", type=int, default=8, help="Engine analysis depth.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    if args.num_positions <= 0:
        raise ValueError("--num_positions must be positive.")
    if args.depth <= 0:
        raise ValueError("--depth must be positive.")
    if args.mode == "pgn" and not args.pgn_path:
        raise ValueError("--pgn_path is required when --mode pgn.")

    logger.info(
        "Dataset generation started | mode=%s | num_positions=%d | depth=%d",
        args.mode,
        args.num_positions,
        args.depth,
    )

    if args.mode == "random":
        positions = generate_random_positions(args.num_positions)
    else:
        positions = generate_positions_from_pgn(args.pgn_path, args.num_positions)
        if len(positions) < args.num_positions:
            logger.warning(
                "PGN yielded %d positions (< requested %d).",
                len(positions),
                args.num_positions,
            )

    if not positions:
        raise RuntimeError("No positions generated. Check mode inputs.")

    progress_every = max(1, min(100, args.num_positions // 10 or 1))

    engine: Optional[chess.engine.SimpleEngine] = None
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        rows = label_positions(
            positions=positions[: args.num_positions],
            engine=engine,
            depth=args.depth,
            progress_every=progress_every,
        )
    finally:
        if engine is not None:
            engine.quit()

    write_csv(rows, args.output_csv)
    logger.info("Wrote %d rows to %s", len(rows), args.output_csv)

    for idx, (fen, eval_value) in enumerate(rows[:3], start=1):
        logger.info("Example %d | fen=%s | eval=%.4f", idx, fen, eval_value)


if __name__ == "__main__":
    main()
