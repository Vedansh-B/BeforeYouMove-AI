"""Inference helpers for chess value model."""

from __future__ import annotations

import chess
import torch

try:
    from deep_learning.board_encoder import encode_board
    from deep_learning.features import extract_global_features
    from deep_learning.model import ChessValueCNN
except ImportError:
    from board_encoder import encode_board
    from features import extract_global_features
    from model import ChessValueCNN


def _infer_input_channels(state: dict) -> int:
    """Infer expected input channel count from a saved state dict."""
    first_weight = state.get("features.0.weight")
    if first_weight is None or first_weight.ndim < 2:
        return 18
    return int(first_weight.shape[1])


def _infer_feature_dim(state: dict) -> int:
    """Infer scalar feature dimension from a saved state dict."""
    first_weight = state.get("feature_head.0.weight")
    if first_weight is None or first_weight.ndim < 2:
        return 24
    return int(first_weight.shape[1])


def load_model(model_path: str = "deep_learning/chess_value_model.pt") -> ChessValueCNN:
    """Load model weights from disk and return an eval-mode model on CPU."""
    state = torch.load(model_path, map_location="cpu")
    model = ChessValueCNN(
        input_channels=_infer_input_channels(state),
        feature_dim=_infer_feature_dim(state),
    )
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Model weights are incompatible with the current hybrid architecture. "
            "Retrain with deep_learning/train.py to regenerate deep_learning/chess_value_model.pt."
        ) from exc
    model.eval()
    return model


@torch.no_grad()
def evaluate_board(board: chess.Board, model: ChessValueCNN) -> float:
    """Evaluate one board and return scalar prediction as float."""
    encoded = encode_board(board)
    scalar_features = extract_global_features(board)
    expected_channels = model.features[0].in_channels
    if encoded.shape[0] != expected_channels:
        encoded = encoded[:expected_channels]
    expected_feature_dim = model.feature_head[0].in_features
    if scalar_features.shape[0] != expected_feature_dim:
        scalar_features = scalar_features[:expected_feature_dim]
    x = torch.from_numpy(encoded).unsqueeze(0).to(torch.float32)
    feat = torch.from_numpy(scalar_features).unsqueeze(0).to(torch.float32)
    out = model(x, feat)
    return float(out.item())


if __name__ == "__main__":
    board = chess.Board()
    try:
        model = load_model()
        value = evaluate_board(board, model)
        print("Starting position prediction:", value)
    except FileNotFoundError:
        print("Model weights not found. Train first with deep_learning/train.py.")
