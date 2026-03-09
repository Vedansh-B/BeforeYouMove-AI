"""Inference helpers for chess value model."""

from __future__ import annotations

import chess
import torch

try:
    from deep_learning.board_encoder import encode_board
    from deep_learning.model import ChessValueCNN
except ImportError:
    from board_encoder import encode_board
    from model import ChessValueCNN


def load_model(model_path: str = "deep_learning/chess_value_model.pt") -> ChessValueCNN:
    """Load model weights from disk and return an eval-mode model on CPU."""
    model = ChessValueCNN()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_board(board: chess.Board, model: ChessValueCNN) -> float:
    """Evaluate one board and return scalar prediction as float."""
    encoded = encode_board(board)
    x = torch.from_numpy(encoded).unsqueeze(0).to(torch.float32)  # (1, 12, 8, 8)
    out = model(x)  # (1, 1)
    return float(out.item())


if __name__ == "__main__":
    board = chess.Board()
    try:
        model = load_model()
        value = evaluate_board(board, model)
        print("Starting position prediction:", value)
    except FileNotFoundError:
        print("Model weights not found. Train first with deep_learning/train.py.")
