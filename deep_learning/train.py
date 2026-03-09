"""Training script for chess value regression."""

from __future__ import annotations

import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

try:
    from deep_learning.dataset import ChessEvalDataset
    from deep_learning.model import ChessValueCNN
except ImportError:
    from dataset import ChessEvalDataset
    from model import ChessValueCNN


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    return running_loss / max(1, count)


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    running_loss = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        count += batch_size

    return running_loss / max(1, count)


def main():
    parser = argparse.ArgumentParser(description="Train chess value model.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV with columns: fen,eval")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dataset = ChessEvalDataset(args.csv)
    n_total = len(dataset)
    if n_total < 2:
        raise ValueError("Need at least 2 rows for train/validation split.")

    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessValueCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Device: {device}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    os.makedirs("deep_learning", exist_ok=True)
    model_path = "deep_learning/chess_value_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to: {model_path}")


if __name__ == "__main__":
    main()
