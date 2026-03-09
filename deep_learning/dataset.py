"""PyTorch dataset for supervised chess evaluation regression."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import List, Tuple

import chess
import torch
from torch.utils.data import Dataset

try:
    from deep_learning.board_encoder import encode_board
except ImportError:
    from board_encoder import encode_board


class ChessEvalDataset(Dataset):
    """Dataset reading rows with columns: fen,eval."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.rows: List[Tuple[str, float]] = []

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fen = row["fen"].strip()
                eval_value = float(row["eval"])
                self.rows.append((fen, eval_value))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        fen, eval_value = self.rows[idx]
        board = chess.Board(fen)
        encoded = encode_board(board)
        x = torch.from_numpy(encoded).to(torch.float32)
        y = torch.tensor([eval_value], dtype=torch.float32)  # shape (1,)
        return x, y


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
        sample_x, sample_y = ds[0]
        print("X shape:", tuple(sample_x.shape))
        print("Y shape:", tuple(sample_y.shape))
        print("Y dtype:", sample_y.dtype)
