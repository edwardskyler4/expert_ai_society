"""Train a randomly initialized Transformer on JSONL conversations."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import IGNORE, PAD, load_records, split_records
from model import Config, TransformerChatbot


def collate(records):
    length = max(len(x) for x, _ in records)
    x = torch.full((len(records), length), PAD, dtype=torch.long)
    y = torch.full_like(x, IGNORE)
    for row, (inputs, labels) in enumerate(records):
        x[row, :len(inputs)] = torch.tensor(inputs)
        y[row, :len(labels)] = torch.tensor(labels)
    return x, y


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, token_count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                ignore_index=IGNORE, reduction="sum",
            )
            count = int((y != IGNORE).sum().item())
            loss = loss_sum / count
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; try reducing the learning rate.")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss_sum.item()
            token_count += count
    return total_loss / token_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/demo.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, threads, and learning rate must be positive.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    config = Config(args.block_size, args.width, args.heads, args.layers, args.dropout)
    records = load_records(args.data, config.block_size)
    training, validation = split_records(records, args.val_fraction, args.seed)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=collate)
    model = TransformerChatbot(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    args.out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | parameters: {parameters:,} | train/validation: {len(training)}/{len(validation)}")
    print("Demo data demonstrates the pipeline; it is too small for general conversation.")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        checkpoint = {
            "format": "byte_transformer_chatbot_v1", "config": asdict(config),
            "model": model.state_dict(), "epoch": epoch,
            "validation_loss": val_loss, "seed": args.seed,
        }
        torch.save(checkpoint, args.out / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(checkpoint, args.out / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:4d} | train {train_loss:.4f} | validation {val_loss:.4f}")
    print(f"Best validation loss: {best:.4f}. Chat with: python chat.py --checkpoint {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
