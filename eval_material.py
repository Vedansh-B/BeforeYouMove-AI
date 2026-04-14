"""Evaluation interface and lightweight chess heuristic evaluator."""

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
    """Lightweight evaluator based on material plus a few chess heuristics."""

    # Piece values in centipawns
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    KING_MIDDLEGAME_TABLE = [
        -50, -40, -40, -35, -35, -40, -40, -50,
        -35, -25, -20, -15, -15, -20, -25, -35,
        -30, -15, -10,  -5,  -5, -10, -15, -30,
        -25, -10,   0,  10,  10,   0, -10, -25,
        -25, -10,   0,  10,  10,   0, -10, -25,
        -30, -15, -10,  -5,  -5, -10, -15, -30,
        -35, -25, -20, -15, -15, -20, -25, -35,
        -50, -40, -40, -35, -35, -40, -40, -50,
    ]

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate position from White's perspective.
        Returns a centipawn score using lightweight strategic heuristics.
        """
        score = self._material_score(board)
        score += self._bishop_pair_bonus(board)
        score += self._mobility_score(board)
        score += self._castling_and_king_safety(board)
        return score

    def _material_score(self, board: chess.Board) -> float:
        score = 0.0

        for piece_type, value in self.PIECE_VALUES.items():
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            score += (white_count - black_count) * value

        return score

    def _bishop_pair_bonus(self, board: chess.Board) -> float:
        bonus = 0.0
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            bonus += 30
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            bonus -= 30
        return bonus

    def _mobility_score(self, board: chess.Board) -> float:
        # Small mobility term to help differentiate equal-material positions.
        turn = board.turn
        our_moves = board.legal_moves.count()
        board.push(chess.Move.null())
        opp_moves = board.legal_moves.count()
        board.pop()

        mobility = our_moves - opp_moves
        return 2.0 * mobility if turn == chess.WHITE else -2.0 * mobility

    def _castling_and_king_safety(self, board: chess.Board) -> float:
        score = 0.0

        if board.has_kingside_castling_rights(chess.WHITE):
            score += 25
        if board.has_queenside_castling_rights(chess.WHITE):
            score += 20
        if board.has_kingside_castling_rights(chess.BLACK):
            score -= 25
        if board.has_queenside_castling_rights(chess.BLACK):
            score -= 20

        if self._is_middlegame(board):
            score += self._king_square_penalty(board, chess.WHITE)
            score -= self._king_square_penalty(board, chess.BLACK)

        return score

    def _is_middlegame(self, board: chess.Board) -> bool:
        non_pawn_material = 0
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            count = len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            non_pawn_material += count * self.PIECE_VALUES[piece_type]
        return non_pawn_material >= 2200

    def _king_square_penalty(self, board: chess.Board, color: chess.Color) -> float:
        king_square = board.king(color)
        if king_square is None:
            return 0.0

        square = king_square if color == chess.WHITE else chess.square_mirror(king_square)
        penalty = self.KING_MIDDLEGAME_TABLE[square]

        if color == chess.WHITE and king_square == chess.E1 and not board.has_castling_rights(chess.WHITE):
            penalty -= 25
        if color == chess.BLACK and king_square == chess.E8 and not board.has_castling_rights(chess.BLACK):
            penalty -= 25

        return float(penalty)
