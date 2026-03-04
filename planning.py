"""Automated planning: bounded-horizon search with heuristic move ordering."""

import chess
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    """Result of a planned line."""
    san_line: str           # Space-separated SAN moves
    uci_line: List[str]     # List of UCI moves for verification
    leaf_fen: str           # FEN at end of line
    leaf_eval: float        # Leaf position evaluation (material)
    delta: float            # leaf_eval - root_eval
    rank: int               # Rank (1=best)


class Planner:
    """Generates and ranks candidate multi-move plans."""

    def __init__(self):
        """Initialize planner."""
        self.nodes_visited = 0
        self.start_time = 0

    def generate_plans(
        self,
        board: chess.Board,
        evaluator,
        depth: int = 4,
        our_branching: int = 8,
        opp_branching: int = 6,
        top_k: int = 10,
    ) -> List[PlanResult]:
        """
        Generate top K plans from current position.
        
        Args:
            board: Current chess position
            evaluator: Evaluation function (e.g., MaterialEvaluator)
            depth: Search depth in ply (moves)
            our_branching: Max moves to consider for our side
            opp_branching: Max moves to consider for opponent
            top_k: Return top K plans
        
        Returns:
            Sorted list of PlanResult objects (best first)
        """
        self.nodes_visited = 0
        self.start_time = time.time()
        root_eval = evaluator.evaluate(board)
        is_white_root = board.turn  # True if White to move at root

        plans = []

        # Generate all candidate first moves for us
        first_moves = self._get_sorted_moves(board, evaluator, limit=our_branching)

        for first_move in first_moves:
            # Get SAN BEFORE pushing
            first_san = board.san(first_move)
            
            board.push(first_move)
            san_moves = [first_san]
            uci_moves = [first_move.uci()]
            
            # Recursively search from depth 1 to (depth-1)
            results = self._search(
                board,
                evaluator,
                depth - 1,
                our_branching,
                opp_branching,
                san_moves,
                uci_moves,
                root_eval,
                is_white_root,
            )

            board.pop()

            for result in results:
                plans.append(result)

        # Sort by evaluation (best first from root side perspective)
        # If White root: higher eval is better, so reverse sort
        # If Black root: lower eval is better, so normal sort
        if is_white_root:
            plans.sort(key=lambda p: p.leaf_eval, reverse=True)
        else:
            plans.sort(key=lambda p: p.leaf_eval)

        # Keep top K
        plans = plans[:top_k]

        # Assign ranks
        for i, plan in enumerate(plans, 1):
            plan.rank = i

        elapsed = time.time() - self.start_time
        logger.info(
            f"Planning complete: {len(plans)} plans, {self.nodes_visited} nodes, "
            f"{elapsed:.2f}s"
        )

        return plans

    def _search(
        self,
        board: chess.Board,
        evaluator,
        depth: int,
        our_branching: int,
        opp_branching: int,
        san_moves: List[str],
        uci_moves: List[str],
        root_eval: float,
        is_white_root: bool,
    ) -> List[PlanResult]:
        """
        Recursive search for plans. Internal helper.
        
        Returns list of PlanResult for all leaf positions reached.
        """
        self.nodes_visited += 1

        # Base case: reached target depth
        if depth == 0:
            leaf_eval = evaluator.evaluate(board)
            delta = leaf_eval - root_eval
            return [
                PlanResult(
                    san_line=" ".join(san_moves),
                    uci_line=uci_moves.copy(),
                    leaf_fen=board.fen(),
                    leaf_eval=leaf_eval,
                    delta=delta,
                    rank=0,
                )
            ]

        # Choose branching limit based on depth parity
        # (simplified: use same limit for both)
        branching_limit = our_branching

        candidate_moves = self._get_sorted_moves(board, evaluator, limit=branching_limit)

        results = []

        for move in candidate_moves:
            # Get SAN before pushing
            move_san = board.san(move)
            
            board.push(move)
            san_moves_extended = san_moves + [move_san]
            uci_moves_extended = uci_moves + [move.uci()]

            results.extend(
                self._search(
                    board,
                    evaluator,
                    depth - 1,
                    our_branching,
                    opp_branching,
                    san_moves_extended,
                    uci_moves_extended,
                    root_eval,
                    is_white_root,
                )
            )

            board.pop()

        return results

    def _get_sorted_moves(
        self, board: chess.Board, evaluator, limit: Optional[int] = None
    ) -> List[chess.Move]:
        """
        Get legal moves sorted by heuristic (captures > checks > promotions > rest).
        Optionally limit to top N by heuristic score.
        """
        moves = list(board.legal_moves)

        # Score each move
        scored_moves = []
        for move in moves:
            score = self._move_heuristic_score(board, move)
            scored_moves.append((score, move))

        # Sort by score (descending)
        scored_moves.sort(key=lambda x: x[0], reverse=True)

        # Extract moves
        sorted_moves = [move for _, move in scored_moves]

        if limit:
            sorted_moves = sorted_moves[:limit]

        return sorted_moves

    def _move_heuristic_score(self, board: chess.Board, move: chess.Move) -> float:
        """
        Score a move for move ordering. Higher = more likely to be good.
        Heuristic: captures > checks > promotions > rest.
        """
        score = 0

        # Captures: high priority
        if board.is_capture(move):
            score += 1000
            # Bonus for capturing valuable pieces
            victim = board.piece_at(move.to_square)
            if victim:
                victim_value = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}.get(
                    victim.piece_type, 0
                )
                score += victim_value * 10

        # Promotions: high priority
        if move.promotion:
            score += 500

        # Checks: medium priority
        board.push(move)
        if board.is_check():
            score += 400
        board.pop()

        return score


def plan_current_position(
    board: chess.Board,
    evaluator,
    depth: int = 4,
    our_branching: int = 8,
    opp_branching: int = 6,
    top_k: int = 10,
) -> List[PlanResult]:
    """
    Convenience function to generate plans from current position.
    
    Args:
        board: Current position
        evaluator: Evaluation function
        depth: Search depth in ply
        our_branching: Branching limit for our moves
        opp_branching: Branching limit for opponent moves
        top_k: Number of top plans to return
    
    Returns:
        List of PlanResult ranked by evaluation
    """
    # Create a copy to avoid mutating the original
    board_copy = board.copy(stack=True)
    
    planner = Planner()
    plans = planner.generate_plans(
        board_copy, evaluator, depth, our_branching, opp_branching, top_k
    )
    
    return plans
