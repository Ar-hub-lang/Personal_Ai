import pytest
import torch
from torch.nn import functional as F

from backend.document_language import normalize_question
from training.document_reader import features
from training.model import DOCUMENT_CONTEXT_LENGTH
from training.tokenizer import ByteTokenizer
from training.train_document_qa import ABSTAIN, LABELS, collate, prepare_batch_inputs, run_epoch


def test_reader_scores_the_last_real_token_in_a_padded_batch():
    class PredictInputToken(torch.nn.Module):
        def forward(self, tokens):
            return 10.0 * F.one_hot(tokens, num_classes=ByteTokenizer.vocab_size)

    batch = collate([
        (torch.tensor([1, 2, ord("C")]), LABELS.index("C")),
        (torch.tensor([ord("N")]), ABSTAIN),
    ])

    score = run_epoch(PredictInputToken(), [batch], "cpu")

    assert score.correct == score.total == 2
    assert score.evidence_correct == score.abstain_correct == 1


def test_document_training_rejects_context_overflow():
    example = {"id": "too-long", "question": "w" * DOCUMENT_CONTEXT_LENGTH,
               "target": "A", "candidates": [{"id": "A", "text": "Evidence."}]}

    with pytest.raises(ValueError, match="too-long.*640"):
        prepare_batch_inputs([example])


def test_reader_matches_common_typos_and_related_terms_without_fuzzy_names_or_numbers():
    assert normalize_question("Waht clg is near Arpit, 247121?") == "what college is near Arpit, 247121?"
    assert features("refrgierators", "refrigerators")[1] == pytest.approx(0.75)
    assert features("cost", "fee")[1] == pytest.approx(0.8)
    assert features("Aarushi", "Aruashi")[1] == 0
    assert features("247121", "247122")[1] == 0


def test_it_spelling_corrects_prompts_and_preserves_literal_identifiers():
    assert normalize_question("How do I disconect from Glboalprotect VPNN?") == "How do I disconnect from GlobalProtect VPN?"
    assert normalize_question("waht is the pasword-reset route for my pritner qeue?") == "what is the password reset route for my printer queue?"
    assert normalize_question("Check `vpnn` and pritner.example.org and case-vpnn-118") == "Check `vpnn` and pritner.example.org and case-vpnn-118"
