import chess

from deep_learning.board_encoder import encode_board
from deep_learning.features import extract_global_features
from deep_learning.model import ChessValueCNN


def test_feature_shapes_match_model_inputs():
    board = chess.Board()
    planes = encode_board(board)
    features = extract_global_features(board)
    model = ChessValueCNN(input_channels=planes.shape[0], feature_dim=features.shape[0])

    assert planes.shape == (18, 8, 8)
    assert features.shape == (24,)
    assert model.features[0].in_channels == 18
    assert model.feature_head[0].in_features == 24
