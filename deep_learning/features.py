"""Handcrafted global features for the chess value model."""

from __future__ import annotations

import chess
import numpy as np


PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
)

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}


def _normalized_piece_count(board: chess.Board, piece_type: chess.PieceType, color: chess.Color) -> float:
    max_counts = {
        chess.PAWN: 8.0,
        chess.KNIGHT: 2.0,
        chess.BISHOP: 2.0,
        chess.ROOK: 2.0,
        chess.QUEEN: 1.0,
    }
    return len(board.pieces(piece_type, color)) / max_counts[piece_type]


def _material_total(board: chess.Board, color: chess.Color) -> float:
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        total += len(board.pieces(piece_type, color)) * value
    return total


def extract_global_features(board: chess.Board) -> np.ndarray:
    """Return normalized scalar features that expose obvious chess structure."""
    features: list[float] = []

    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_TYPES:
            features.append(_normalized_piece_count(board, piece_type, color))

    white_material = _material_total(board, chess.WHITE)
    black_material = _material_total(board, chess.BLACK)
    features.append((white_material - black_material) / 39.0)
    features.append(white_material / 39.0)
    features.append(black_material / 39.0)

    features.append(1.0 if board.turn == chess.WHITE else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.is_check() else 0.0)

    our_mobility = board.legal_moves.count()
    board.push(chess.Move.null())
    opp_mobility = board.legal_moves.count()
    board.pop()
    features.append(our_mobility / 64.0)
    features.append(opp_mobility / 64.0)
    features.append((our_mobility - opp_mobility) / 64.0)

    occupied = len(board.piece_map())
    features.append(occupied / 32.0)
    features.append(board.fullmove_number / 100.0)

    return np.asarray(features, dtype=np.float32)
