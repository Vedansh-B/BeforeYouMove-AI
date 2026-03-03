"""Evaluation interface and material-only evaluator."""

from abc import ABC, abstractmethod
import chess


class Evaluator(ABC):
    """Abstract interface for position evaluators."""

    @abstractmethod
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position from White's perspective.
        Positive = White advantage, Negative = Black advantage.
        Returned in centipawns (cp).
        """
        pass


class MaterialEvaluator(Evaluator):
    """Simple evaluation based on material count."""

    # Piece values in centipawns
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position from White's perspective.
        Returns material count difference in centipawns.
        """
        score = 0.0

        for piece_type, value in self.PIECE_VALUES.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += (white_count - black_count) * value

        return score
