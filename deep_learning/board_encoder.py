"""Board encoding utilities for chess value regression."""

from __future__ import annotations

import chess
import numpy as np

# Plane order (12, 8, 8):
# 0 white pawns
# 1 white knights
# 2 white bishops
# 3 white rooks
# 4 white queens
# 5 white king
# 6 black pawns
# 7 black knights
# 8 black bishops
# 9 black rooks
# 10 black queens
# 11 black king
PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a python-chess board into 12 binary planes.

    Returns:
        np.ndarray of shape (12, 8, 8), dtype float32.
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        rank = chess.square_rank(square)  # 0..7 (rank 1..8)
        file = chess.square_file(square)  # 0..7 (file a..h)
        row = 7 - rank  # row 0 at top (8th rank)
        col = file
        planes[plane_idx, row, col] = 1.0

    return planes


if __name__ == "__main__":
    test_board = chess.Board()
    encoded = encode_board(test_board)
    print("Encoded shape:", encoded.shape)
    print("dtype:", encoded.dtype)
    print("Total ones:", int(encoded.sum()))
