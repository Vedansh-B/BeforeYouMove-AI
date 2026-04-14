"""PyTorch dataset for supervised chess evaluation regression."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import chess
import torch
from torch.utils.data import Dataset

try:
    from deep_learning.board_encoder import encode_board
    from deep_learning.features import extract_global_features
except ImportError:
    from board_encoder import encode_board
    from features import extract_global_features


class ChessEvalDataset(Dataset):
    """Dataset reading rows with columns: fen,eval."""

    def __init__(self, csv_paths: str | Sequence[str], augment_mirror: bool = False):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]

        self.csv_paths = list(csv_paths)
        self.augment_mirror = augment_mirror
        self.rows: List[Tuple[str, float]] = []

        for csv_path in self.csv_paths:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fen = row["fen"].strip()
                    eval_value = float(row["eval"])
                    self.rows.append((fen, eval_value))

                    if augment_mirror:
                        board = chess.Board(fen)
                        mirrored = board.transform(chess.flip_horizontal)
                        self.rows.append((mirrored.fen(), eval_value))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        fen, eval_value = self.rows[idx]
        board = chess.Board(fen)
        encoded = encode_board(board)
        global_features = extract_global_features(board)
        x = torch.from_numpy(encoded).to(torch.float32)
        features = torch.from_numpy(global_features).to(torch.float32)
        y = torch.tensor([eval_value], dtype=torch.float32)  # shape (1,)
        return x, features, y


if __name__ == "__main__":
    # Small smoke test with a temporary CSV.
    content = """fen,eval
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0.0
rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2,0.1
"""
    with tempfile.TemporaryDirectory() as td:
        tmp_csv = Path(td) / "tiny.csv"
        tmp_csv.write_text(content, encoding="utf-8")

        ds = ChessEvalDataset(str(tmp_csv))
        print("Dataset size:", len(ds))
        sample_x, sample_features, sample_y = ds[0]
        print("X shape:", tuple(sample_x.shape))
        print("Feature shape:", tuple(sample_features.shape))
        print("Y shape:", tuple(sample_y.shape))
        print("Y dtype:", sample_y.dtype)
