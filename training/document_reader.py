"""A shared TinyGPT scorer for document evidence and an abstention option."""

import json
import re
import unicodedata
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from backend.document_language import RELATED_WORDS, WORD, close_spelling, normalize_question
from training.model import DOCUMENT_CONTEXT_LENGTH, TinyGPT
from training.tokenizer import ByteTokenizer


CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "tiny-gpt-document-reader.pt"
PAIR_WINDOW = 256
YEAR = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
NUMBER = re.compile(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million)\b", re.I)
STOP = set("a an are as at be by can could did do does for from had has have how in is it me of on or please tell the to was were what when where which who will with would you".split())


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=[^\W\d_])['’]s\b", "", text)
    return WORD.findall(text)


def stem(word: str) -> str:
    for suffix in ("ing", "ied", "ed", "es", "s"):
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


ALIAS_STEMS = {stem(word): {stem(other) for other in group} for group in RELATED_WORDS for word in group}


def features(question: str, passage: str) -> list[float]:
    question = normalize_question(question)
    question_words = words(question)
    passage_words = words(passage)
    query = {word for word in question_words if word not in STOP}
    source = set(passage_words)
    shared = query & source
    query_stems = {stem(word) for word in query}
    source_stems = {stem(word) for word in source}
    aliases = {word for word in query_stems - source_stems if ALIAS_STEMS.get(word, set()) & source_stems}
    passage_spelling = {word for word in WORD.findall(passage) if word.isalpha() and word.islower() and len(word) >= 5}
    fuzzy = {stem(word) for word in WORD.findall(question)
             if word.isalpha() and word.islower() and word in query and stem(word) not in (source_stems | aliases)
             and len(word) >= 6 and any(close_spelling(word, item) for item in passage_spelling)}
    query_bigrams = {(a, b) for a, b in zip(question_words, question_words[1:]) if a not in STOP and b not in STOP}
    source_bigrams = set(zip(passage_words, passage_words[1:]))
    asked_year = "year" in query or "date" in query
    asked_number = re.search(r"\b(?:how (?:many|much|long)|number of|percent|price|speed)\b", question, re.I)
    return [
        len(shared) / max(1, len(query)),
        (len(query_stems & source_stems) + 0.8 * len(aliases) + 0.75 * len(fuzzy)) / max(1, len(query_stems)),
        len(shared) / max(1, len(source)) ** 0.5,
        len(query_bigrams & source_bigrams) / max(1, len(query_bigrams)),
        float(bool(asked_year and YEAR.search(passage))),
        float(bool(asked_number and NUMBER.search(passage))),
        float(bool(asked_year and not YEAR.search(passage))),
        float(bool(asked_number and not NUMBER.search(passage))),
    ]


def encode_pair(question: str, passage: str) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = ByteTokenizer()
    question_bytes = tokenizer.encode(f"Q: {question}\n")[:96]
    separator = tokenizer.encode("\nEvidence:\n")
    room = PAIR_WINDOW - 1 - len(question_bytes) - len(separator)
    passage_bytes = tokenizer.encode(passage)
    if len(passage_bytes) > room:
        head = room * 2 // 3
        passage_bytes = passage_bytes[:head] + [ord(" ")] + passage_bytes[-(room - head - 1):]
    return (torch.tensor([tokenizer.bos_id, *question_bytes, *separator, *passage_bytes]),
            torch.tensor(features(question, passage)))


def collate_pairs(pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    tokens, overlap = zip(*pairs)
    return (pad_sequence(tokens, batch_first=True), torch.tensor([len(row) - 1 for row in tokens]),
            torch.stack(overlap))


class DocumentReader(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tiny_gpt = TinyGPT(context_length=DOCUMENT_CONTEXT_LENGTH)
        self.overlap = nn.Linear(8, 1)
        with torch.no_grad():
            self.overlap.weight.copy_(torch.tensor([[2.0, 2.0, 0.4, 0.6, 0.4, 0.3, -0.6, -0.5]]))
            self.overlap.bias.fill_(-2.0)

    def forward(self, tokens: torch.Tensor, last: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
        logits = self.tiny_gpt(tokens)[torch.arange(len(tokens), device=tokens.device), last]
        learned_score = (logits[:, ord("Y")] - logits[:, ord("N")]).tanh()
        return self.overlap(overlap).squeeze(-1) + 0.5 * learned_score

    @torch.inference_mode()
    def rank(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        device = self.overlap.weight.device
        batch = collate_pairs([encode_pair(question, passage) for passage in passages])
        return self(*(tensor.to(device) for tensor in batch)).cpu().tolist()


def reader_configuration(model: DocumentReader) -> dict:
    transformer = model.tiny_gpt
    layer = transformer.blocks[0]
    return {
        "version": 1, "architecture": "DocumentReader",
        "vocab_size": transformer.token_embedding.num_embeddings,
        "context_length": transformer.context_length,
        "width": transformer.token_embedding.embedding_dim,
        "heads": layer.self_attn.num_heads, "layers": len(transformer.blocks),
        "dropout": layer.dropout.p, "feature_count": model.overlap.in_features,
    }


def load_reader(path: Path = CHECKPOINT) -> tuple[DocumentReader, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = DocumentReader()
    sidecar = path.with_suffix(".json")
    if sidecar.is_file():
        config = json.loads(sidecar.read_text(encoding="utf-8"))
        if config != reader_configuration(model):
            raise ValueError("The archived reader configuration does not match this model architecture.")
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved


def sentence_spans(text: str) -> list[tuple[int, int]]:
    start = 0
    spans = []
    for match in re.finditer(r"(?<=[.!?])\s+(?=[\"“(]?[A-Z0-9])", text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [(left, right) for left, right in spans if right - left >= 20]
