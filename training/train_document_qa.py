"""Train TinyGPT to select a supplied evidence passage or abstain."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from training.document_qa_data import CHOICES, NO_ANSWER, SPLITS, format_prompt, read_split
from training.model import DOCUMENT_CONTEXT_LENGTH, TinyGPT
from training.tokenizer import ByteTokenizer
from training.train import save_checkpoint


LABELS = (*CHOICES, NO_ANSWER)
ABSTAIN = LABELS.index(NO_ANSWER)
DEFAULT_CHECKPOINT = Path("models/tiny-gpt-document-qa.pt")


@dataclass
class Score:
    total_loss: float = 0
    correct: int = 0
    total: int = 0
    evidence_correct: int = 0
    evidence_total: int = 0
    abstain_correct: int = 0
    abstain_total: int = 0

    @property
    def loss(self) -> float:
        return self.total_loss / self.total

    def summary(self) -> str:
        return (f"{self.correct}/{self.total} correct, loss {self.loss:.4f} "
                f"(evidence {self.evidence_correct}/{self.evidence_total}, "
                f"NO_ANSWER {self.abstain_correct}/{self.abstain_total})")


def prepare_batch_inputs(examples: list[dict]) -> list[tuple[torch.Tensor, int]]:
    tokenizer = ByteTokenizer()
    rows = []
    for example in examples:
        tokens = [tokenizer.bos_id, *tokenizer.encode(format_prompt(example) + " ")]
        if len(tokens) > DOCUMENT_CONTEXT_LENGTH:
            raise ValueError(f"{example['id']} needs {len(tokens)} tokens. The context is {DOCUMENT_CONTEXT_LENGTH}.")
        target = LABELS.index(example["target"])
        rows.append((torch.tensor(tokens), target))
    if not rows:
        raise ValueError("A document QA split is empty.")
    return rows


def collate(batch):
    prompts, labels = zip(*batch, strict=True)
    lengths = torch.tensor([len(prompt) for prompt in prompts])
    return pad_sequence(prompts, batch_first=True), lengths, torch.tensor(labels)


def run_epoch(model, loader, device, optimizer=None) -> Score:
    training = optimizer is not None
    model.train(training)
    score = Score()
    # Each label has a reserved single-byte choice at the end of the prompt.
    label_tokens = torch.tensor([ord(label[0]) for label in LABELS], device=device)
    with torch.set_grad_enabled(training):
        for prompts, lengths, targets in loader:
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(prompts.to(device))
            batch_indices = torch.arange(len(prompts), device=device)
            # Read the last real prompt token. Right-padding comes after this point.
            choices = logits[batch_indices, lengths.to(device) - 1].index_select(-1, label_tokens)
            loss = F.cross_entropy(choices, targets)
            if not math.isfinite(loss.item()):
                raise ValueError("Document QA loss became non-finite.")
            if training:
                loss.backward()
                optimizer.step()

            correct = choices.detach().argmax(dim=-1).eq(targets)
            abstain = targets.eq(ABSTAIN)
            score.total_loss += loss.item() * len(prompts)
            score.total += len(prompts)
            score.correct += int(correct.sum())
            score.evidence_correct += int((correct & ~abstain).sum())
            score.evidence_total += int((~abstain).sum())
            score.abstain_correct += int((correct & abstain).sum())
            score.abstain_total += int(abstain.sum())
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--evaluate-only", action="store_true", help="Load the checkpoint and score it.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Run with --device cpu.")

    torch.manual_seed(42)
    try:
        examples = {split: read_split(split) for split in SPLITS}
        loaders = {
            split: DataLoader(prepare_batch_inputs(rows), batch_size=args.batch_size,
                              shuffle=split == "train", collate_fn=collate)
            for split, rows in examples.items()
        }
        model = TinyGPT(context_length=DOCUMENT_CONTEXT_LENGTH).to(args.device)
        if not args.evaluate_only:
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            baseline = run_epoch(model, loaders["validation"], args.device)
            print(f"Untrained validation: {baseline.summary()}", flush=True)
            best = None
            for epoch in range(1, args.epochs + 1):
                trained = run_epoch(model, loaders["train"], args.device, optimizer)
                validation = run_epoch(model, loaders["validation"], args.device)
                print(f"Epoch {epoch:02d}: train {trained.correct}/{trained.total}; "
                      f"validation {validation.summary()}", flush=True)
                rank = (validation.correct, validation.evidence_correct, -validation.loss)
                if best is None or rank > best:
                    save_checkpoint(args.checkpoint, model, epoch, validation.loss)
                    best = rank

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})", flush=True)
        for split in ("validation", "test"):
            score = run_epoch(model, loaders[split], args.device)
            print(f"{split}: {score.summary()}", flush=True)
            print(f"  Always choosing NO_ANSWER scores {score.abstain_total}/{score.total}.", flush=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
