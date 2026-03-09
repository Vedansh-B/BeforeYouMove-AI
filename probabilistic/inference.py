"""Probabilistic first-move inference with a strong-opponent rollout model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import logging
import math
import random
import time

import chess

from planning import Planner

logger = logging.getLogger(__name__)

PAWN_CP = 100.0
DEFAULT_OPPONENT_RANK_WEIGHTS = (0.6, 0.3, 0.1)


@dataclass
class MoveProbResult:
    """Aggregated probabilistic estimate for one candidate first move."""

    san: str
    uci: str
    success_prob: float
    avg_leaf_score: float
    avg_delta: float
    simulations: int


def _to_root_perspective(raw_eval_cp: float, root_turn: bool) -> float:
    """
    Convert raw evaluator score (White-centric cp) to root-side perspective.

    If root is White, higher cp is better for root.
    If root is Black, lower raw cp is better for root, so we flip sign.
    """
    return raw_eval_cp if root_turn == chess.WHITE else -raw_eval_cp


def _ranked_weight_distribution(size: int, rank_weights: Sequence[float]) -> List[float]:
    """Build a normalized probability distribution by rank."""
    if size <= 0:
        return []

    usable = list(rank_weights[:size])
    if len(usable) < size:
        usable.extend([0.0] * (size - len(usable)))

    total = sum(max(0.0, w) for w in usable)
    if total <= 0:
        return [1.0 / size] * size

    return [max(0.0, w) / total for w in usable]


def _get_top_moves_by_eval(
    board: chess.Board,
    evaluator,
    top_k: int,
    perspective_turn: bool,
) -> List[Tuple[chess.Move, float]]:
    """
    Score legal moves by one-ply eval and return best moves for a side perspective.

    `perspective_turn` says which side the ranking should optimize for.
    """
    scored: List[Tuple[chess.Move, float]] = []

    for move in board.legal_moves:
        board.push(move)
        raw = evaluator.evaluate(board)
        perspective_eval = _to_root_perspective(raw, perspective_turn)
        scored.append((move, perspective_eval))
        board.pop()

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(1, top_k)]


def _choose_best_move_for_root(
    board: chess.Board,
    evaluator,
    root_turn: bool,
) -> Optional[chess.Move]:
    """Deterministic policy for root-side plies: choose best one-ply eval move."""
    if board.is_game_over():
        return None

    best_move = None
    best_score = -math.inf

    for move in board.legal_moves:
        board.push(move)
        raw = evaluator.evaluate(board)
        score = _to_root_perspective(raw, root_turn)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def sample_strong_opponent_reply(
    board: chess.Board,
    evaluator,
    top_k: int = 3,
    rank_weights: Sequence[float] = DEFAULT_OPPONENT_RANK_WEIGHTS,
) -> chess.Move:
    """
    Sample an opponent reply from the strongest candidates.

    Opponent strength model:
    1) score legal replies from opponent perspective
    2) keep top-K strongest replies
    3) sample with rank-weighted probabilities (best move favored)
    """
    if board.is_game_over():
        raise ValueError("No legal reply in a game-over position.")

    opponent_turn = board.turn
    top_scored = _get_top_moves_by_eval(
        board=board,
        evaluator=evaluator,
        top_k=max(1, top_k),
        perspective_turn=opponent_turn,
    )

    moves = [move for move, _ in top_scored]
    probs = _ranked_weight_distribution(len(moves), rank_weights)
    return random.choices(moves, weights=probs, k=1)[0]


def run_rollout_after_first_move(
    board: chess.Board,
    first_move: chess.Move,
    evaluator,
    horizon: int,
    opponent_top_k: int,
    root_turn: bool,
) -> float:
    """
    Roll forward from one chosen first move and return root-perspective leaf score.

    The board passed in is copied internally, so caller board is never mutated.
    """
    rollout = board.copy(stack=True)
    rollout.push(first_move)

    for _ in range(max(0, horizon)):
        if rollout.is_game_over():
            break

        if rollout.turn == root_turn:
            move = _choose_best_move_for_root(rollout, evaluator, root_turn=root_turn)
        else:
            move = sample_strong_opponent_reply(
                rollout,
                evaluator,
                top_k=opponent_top_k,
            )

        if move is None:
            break
        rollout.push(move)

    leaf_raw = evaluator.evaluate(rollout)
    return _to_root_perspective(leaf_raw, root_turn) / PAWN_CP


def estimate_first_move_successes(
    board: chess.Board,
    evaluator,
    horizon: int = 4,
    top_k: int = 10,
    opponent_top_k: int = 3,
    simulations: int = 50,
    success_threshold: float = 0.5,
) -> List[MoveProbResult]:
    """
    Estimate P(success | first_move = m, strong_opponent) for top-N first moves.

    Success definition:
        leaf_score_root - root_score_root >= success_threshold
    with scores measured in pawns.
    """
    start = time.time()
    root_turn = board.turn
    root_score = _to_root_perspective(evaluator.evaluate(board), root_turn) / PAWN_CP

    planner = Planner()
    candidate_moves = planner._get_sorted_moves(  # Reuse existing planning heuristic.
        board=board,
        evaluator=evaluator,
        limit=max(1, top_k),
    )

    logger.info(
        "Probabilistic inference: candidates=%d, simulations_per_move=%d, horizon=%d, opponent_top_k=%d, threshold=%+.2f pawns",
        len(candidate_moves),
        simulations,
        horizon,
        opponent_top_k,
        success_threshold,
    )

    results: List[MoveProbResult] = []
    safe_simulations = max(1, simulations)

    for first_move in candidate_moves:
        first_san = board.san(first_move)
        leaf_scores: List[float] = []
        success_count = 0

        for _ in range(safe_simulations):
            leaf_score = run_rollout_after_first_move(
                board=board,
                first_move=first_move,
                evaluator=evaluator,
                horizon=horizon,
                opponent_top_k=opponent_top_k,
                root_turn=root_turn,
            )
            leaf_scores.append(leaf_score)
            if (leaf_score - root_score) >= success_threshold:
                success_count += 1

        avg_leaf = sum(leaf_scores) / len(leaf_scores)
        avg_delta = avg_leaf - root_score
        success_prob = success_count / len(leaf_scores)

        results.append(
            MoveProbResult(
                san=first_san,
                uci=first_move.uci(),
                success_prob=success_prob,
                avg_leaf_score=avg_leaf,
                avg_delta=avg_delta,
                simulations=len(leaf_scores),
            )
        )

    results.sort(key=lambda r: (r.success_prob, r.avg_delta, r.avg_leaf_score), reverse=True)

    elapsed = time.time() - start
    if results:
        best = results[0]
        logger.info(
            "Probabilistic inference done in %.2fs | best=%s (%s) P=%.2f",
            elapsed,
            best.san,
            best.uci,
            best.success_prob,
        )
    else:
        logger.info("Probabilistic inference done in %.2fs | no legal candidates", elapsed)

    return results
