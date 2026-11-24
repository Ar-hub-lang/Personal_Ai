import argparse
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from training.model import TinyGPT
from training.tokenizer import ByteTokenizer


def read_examples(path: Path) -> list[str]:
    examples = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples:
        raise ValueError(f"{path} must contain at least one nonempty line.")
    return examples


def make_windows(examples: list[str], tokenizer: ByteTokenizer, context: int) -> list[torch.Tensor]:
    windows = []
    for example in examples:
        tokens = tokenizer.encode(example, add_boundaries=True)
        for start in range(0, len(tokens) - 1, context):
            windows.append(torch.tensor(tokens[start : start + context + 1]))
    return windows


def collate(windows):
    inputs = pad_sequence([window[:-1] for window in windows], batch_first=True)
    targets = pad_sequence([window[1:] for window in windows], batch_first=True, padding_value=-100)
    return inputs, targets


def run_epoch(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            count = int((targets != -100).sum())
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
            if not math.isfinite(loss.item()):
                raise ValueError("Loss became non-finite. Try a lower learning rate.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * count
            total_tokens += count
    return total_loss / total_tokens


def save_checkpoint(path: Path, model, epoch: int, validation_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(
            {
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "validation_loss": validation_loss,
            },
            temporary,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local TinyGPT on two text files.")
    parser.add_argument("--train", type=Path, default=Path("training/train.txt"))
    parser.add_argument("--validation", type=Path, default=Path("training/validation.txt"))
    parser.add_argument("--output", type=Path, default=Path("models/tiny-gpt.pt"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Run with --device cpu.")
    if args.output.resolve() in {args.train.resolve(), args.validation.resolve()}:
        parser.error("The checkpoint path must differ from both input files.")
    try:
        train = read_examples(args.train)
        validation = read_examples(args.validation)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if set(train) & set(validation):
        parser.error("A sentence appears in both files. Keep validation examples separate.")

    print(f"Examples: {len(train)} training, {len(validation)} validation.", flush=True)
    if min(len(train), len(validation)) < 10:
        print("This dataset is small enough for a pipeline check; add more examples to judge generalization.", flush=True)
    torch.manual_seed(42)
    model = TinyGPT().to(args.device)
    tokenizer = ByteTokenizer()
    train_loader = DataLoader(
        make_windows(train, tokenizer, model.context_length), batch_size=16, shuffle=True, collate_fn=collate
    )
    validation_loader = DataLoader(
        make_windows(validation, tokenizer, model.context_length), batch_size=16, collate_fn=collate
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    baseline = run_epoch(model, validation_loader, args.device)
    print(f"Untrained validation loss: {baseline:.4f}", flush=True)
    best_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, args.device, optimizer)
        validation_loss = run_epoch(model, validation_loader, args.device)
        print(f"Epoch {epoch:02d}: train {train_loss:.4f} | validation {validation_loss:.4f}", flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(args.output, model, epoch, best_loss)
    print(f"Best trained checkpoint: {args.output} (validation loss {best_loss:.4f})")


if __name__ == "__main__":
    main()
