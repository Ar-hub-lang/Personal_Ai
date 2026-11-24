import argparse

import torch
from torch.nn import functional as F

from training.model import DOCUMENT_CONTEXT_LENGTH, TEXT_CONTEXT_LENGTH, TinyGPT
from training.tokenizer import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the tokenizer and untrained Transformer.")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--document-qa", action="store_true", help="Check the longer context against the PDF dataset.")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable. Run with --device cpu.")

    tokenizer = ByteTokenizer()
    text = "Hello, नमस्ते!"
    token_ids = tokenizer.encode(text, add_boundaries=True)
    assert tokenizer.decode(token_ids) == text

    context_length = DOCUMENT_CONTEXT_LENGTH if args.document_qa else TEXT_CONTEXT_LENGTH
    if args.document_qa:
        from training.document_qa_data import SPLITS, encode_example, read_split

        try:
            examples = [example for split in SPLITS for example in read_split(split)]
            token_ids = max((encode_example(example, tokenizer) for example in examples), key=len)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if len(token_ids) > context_length:
            parser.error(f"The longest document example needs {len(token_ids)} tokens, but the context is {context_length}.")
        print(f"Document examples checked: {len(examples)}; longest complete example: {len(token_ids)} tokens.")

    tokens = torch.tensor([token_ids], device=args.device)
    model = TinyGPT(context_length=context_length, dropout=0).to(args.device)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    logits = model(inputs)
    assert logits.shape == (*inputs.shape, tokenizer.vocab_size)

    changed_inputs = inputs.clone()
    changed_inputs[:, -1] = (changed_inputs[:, -1] + 1) % tokenizer.vocab_size
    with torch.no_grad():
        torch.testing.assert_close(logits[:, :-1], model(changed_inputs)[:, :-1])
        if args.document_qa:
            full_window = F.pad(tokens, (0, context_length - tokens.shape[1]), value=ord(" "))
            assert model(full_window).shape == (1, context_length, tokenizer.vocab_size)

    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    loss.backward()
    assert torch.isfinite(loss).item()
    assert model.output.weight.grad is not None
    assert torch.isfinite(model.output.weight.grad).all().item()

    print(f"Tokenizer round trip, causal attention, and gradients checked on {args.device}.")
    print(f"Context: {model.context_length} byte tokens.")
    print(f"Vocabulary: {tokenizer.vocab_size}; parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
