"""Train the shared TinyGPT evidence reader on PDF examples and SQuAD 2.0 sentences."""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from training.document_evidence_examples import document_evidence_examples
from training.document_qa_data import CHOICES, NO_ANSWER, read_split
from training.document_question_variants import variant_rows
from training.document_reader import CHECKPOINT, DocumentReader, collate_pairs, encode_pair, features, sentence_spans
from training.resume_skill_examples import resume_skill_examples
from training.train import save_checkpoint
from training.train_document_qa import Score


SQUAD_FILE = Path("models/training-data/squad-v2-train.json")
ABSTAIN = len(CHOICES)


def squad_examples(path: Path, limit: int) -> list[tuple[str, list[str], int]]:
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    rng = random.Random(42)
    rng.shuffle(data)
    wanted_answerable = limit * 3 // 5
    wanted_missing = limit - wanted_answerable
    examples = []
    for article in data:
        paragraphs = list(article["paragraphs"])
        rng.shuffle(paragraphs)
        for paragraph in paragraphs:
            context = paragraph["context"]
            spans = sentence_spans(context)
            sentences = [context[left:right].strip() for left, right in spans]
            if len(set(sentences)) < 4:
                continue
            questions = list(paragraph["qas"])
            rng.shuffle(questions)
            for example in questions:
                question = example["question"]
                if example["is_impossible"]:
                    if not wanted_missing:
                        continue
                    ranked = sorted(set(sentences), key=lambda text: (-features(question, text)[1], text))
                    candidates = ranked[:4]
                    target = ABSTAIN
                    wanted_missing -= 1
                else:
                    if not wanted_answerable or not example["answers"]:
                        continue
                    answer = example["answers"][0]
                    start = answer["answer_start"]
                    end = start + len(answer["text"])
                    gold = next((sentence for (left, right), sentence in zip(spans, sentences)
                                 if left <= start and end <= right), None)
                    if not gold:
                        continue
                    distractors = {sentence for sentence in sentences if answer["text"].casefold() not in sentence.casefold()}
                    if len(distractors) < 3:
                        continue
                    ranked = sorted(distractors, key=lambda text: (-features(question, text)[1], text))
                    candidates = [gold, *ranked[:3]]
                    rng.shuffle(candidates)
                    target = candidates.index(gold)
                    wanted_answerable -= 1
                examples.append((question, candidates, target))
                break
            if not wanted_answerable and not wanted_missing:
                return examples
    raise ValueError(f"SQuAD yielded only {len(examples)} of {limit} requested examples.")


def pdf_examples(split: str) -> list[tuple[str, list[str], int]]:
    rows = []
    for example in read_split(split):
        question = " ".join(part for part in (example["question"], example.get("context")) if part)
        labels = [item["id"] for item in example["candidates"]]
        target = ABSTAIN if example["target"] == NO_ANSWER else labels.index(example["target"])
        rows.append((question, [item["text"] for item in example["candidates"]], target))
    return rows


def make_loader(rows, batch_size, shuffle=False):
    encoded = [([encode_pair(question, passage) for passage in passages], target)
               for question, passages, target in rows]

    def collate(batch):
        tokens = collate_pairs([pair for pairs, _ in batch for pair in pairs])
        return (*tokens, torch.tensor([target for _, target in batch]))

    return DataLoader(encoded, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def run_epoch(model, loader, device, optimizer=None) -> Score:
    model.train(optimizer is not None)
    score = Score()
    with torch.set_grad_enabled(optimizer is not None):
        for tokens, last, overlap, targets in loader:
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            evidence = model(tokens.to(device), last.to(device), overlap.to(device)).reshape(len(targets), ABSTAIN)
            logits = torch.cat((evidence, torch.zeros_like(evidence[:, :1])), dim=-1)
            loss = F.cross_entropy(logits, targets)
            if not math.isfinite(loss.item()):
                raise ValueError("Evidence reader loss became non-finite.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            correct = logits.detach().argmax(-1).eq(targets)
            abstain = targets.eq(ABSTAIN)
            score.total_loss += loss.item() * len(targets)
            score.total += len(targets)
            score.correct += int(correct.sum())
            score.evidence_correct += int((correct & ~abstain).sum())
            score.evidence_total += int((~abstain).sum())
            score.abstain_correct += int((correct & abstain).sum())
            score.abstain_total += int(abstain.sum())
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squad", type=Path, default=SQUAD_FILE)
    parser.add_argument("--sample-size", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--initialize-from", type=Path, help="Start with an existing reader checkpoint.")
    parser.add_argument("--question-variants", action="store_true", help="Train and test with typos and alternate phrasing.")
    parser.add_argument("--resume-skills", action="store_true", help="Include fictional résumé skills questions.")
    parser.add_argument("--evidence-examples", action="store_true", help="Include evidence and hard-negative equipment records.")
    args = parser.parse_args()
    if args.sample_size < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("Sample size, epochs and batch size must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Run with --device cpu.")

    torch.manual_seed(42)
    try:
        validation_rows = pdf_examples("validation")
        validation = make_loader(validation_rows, args.batch_size)
        validation_variants = make_loader(variant_rows(validation_rows, seed=91), args.batch_size) if args.question_variants else None
        validation_skills = (make_loader(resume_skill_examples("validation", ABSTAIN), args.batch_size)
                             if args.resume_skills else None)
        validation_evidence = (make_loader(document_evidence_examples("validation", ABSTAIN), args.batch_size)
                               if args.evidence_examples else None)
        model = DocumentReader().to(args.device)
        if not args.evaluate_only:
            if args.initialize_from:
                initial = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
                model.load_state_dict(initial["model_state"])
            extra = squad_examples(args.squad, args.sample_size)
            local = pdf_examples("train")
            variants = variant_rows(extra + local, seed=47) if args.question_variants else []
            skills = resume_skill_examples("train", ABSTAIN) if args.resume_skills else []
            evidence = document_evidence_examples("train", ABSTAIN) if args.evidence_examples else []
            train = make_loader(extra + local * 2 + variants + skills + evidence, args.batch_size, shuffle=True)
            print(f"Training on {len(extra)} reading examples, {len(local)} PDF examples and "
                  f"{len(variants)} question variants, plus {len(skills)} résumé skills and "
                  f"{len(evidence)} equipment evidence examples.", flush=True)
            print(f"Before training: validation {run_epoch(model, validation, args.device).summary()}", flush=True)
            if validation_variants is not None:
                print(f"Before training: variants {run_epoch(model, validation_variants, args.device).summary()}", flush=True)
            optimizer = torch.optim.AdamW([
                {"params": model.tiny_gpt.parameters(), "lr": 8e-5 if args.initialize_from else 2e-4},
                {"params": model.overlap.parameters(), "lr": 3e-3 if args.initialize_from else 1e-2},
            ], weight_decay=0.03)
            best = None
            for epoch in range(1, args.epochs + 1):
                learned = run_epoch(model, train, args.device, optimizer)
                checked = run_epoch(model, validation, args.device)
                print(f"Epoch {epoch}: train {learned.correct}/{learned.total}; validation {checked.summary()}", flush=True)
                scores = [checked]
                if validation_variants is not None:
                    variant = run_epoch(model, validation_variants, args.device)
                    print(f"  Validation variants: {variant.summary()}", flush=True)
                    scores.append(variant)
                if validation_skills is not None:
                    skill_score = run_epoch(model, validation_skills, args.device)
                    print(f"  Validation résumé skills: {skill_score.summary()}", flush=True)
                    scores.append(skill_score)
                if validation_evidence is not None:
                    evidence_score = run_epoch(model, validation_evidence, args.device)
                    print(f"  Validation equipment evidence: {evidence_score.summary()}", flush=True)
                    scores.append(evidence_score)
                rank = (sum(score.correct for score in scores), sum(score.abstain_correct for score in scores), checked.correct,
                        sum(score.evidence_correct for score in scores), -sum(score.loss for score in scores))
                if best is None or rank > best:
                    save_checkpoint(args.checkpoint, model, epoch, checked.loss)
                    best = rank

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Checkpoint {args.checkpoint} (epoch {checkpoint['epoch']})", flush=True)
        print(f"Validation: {run_epoch(model, validation, args.device).summary()}", flush=True)
        if validation_variants is not None:
            print(f"Validation variants: {run_epoch(model, validation_variants, args.device).summary()}", flush=True)
        test_rows = pdf_examples("test")
        test = make_loader(test_rows, args.batch_size)
        print(f"Test: {run_epoch(model, test, args.device).summary()}", flush=True)
        if args.question_variants:
            test_variants = make_loader(variant_rows(test_rows, seed=139), args.batch_size)
            print(f"Test variants: {run_epoch(model, test_variants, args.device).summary()}", flush=True)
        if validation_skills is not None:
            print(f"Validation résumé skills: {run_epoch(model, validation_skills, args.device).summary()}", flush=True)
            test_skills = make_loader(resume_skill_examples("test", ABSTAIN), args.batch_size)
            print(f"Test résumé skills: {run_epoch(model, test_skills, args.device).summary()}", flush=True)
        if validation_evidence is not None:
            print(f"Validation equipment evidence: {run_epoch(model, validation_evidence, args.device).summary()}", flush=True)
            test_evidence = make_loader(document_evidence_examples("test", ABSTAIN), args.batch_size)
            print(f"Test equipment evidence: {run_epoch(model, test_evidence, args.device).summary()}", flush=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
