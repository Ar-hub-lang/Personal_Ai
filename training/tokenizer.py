class ByteTokenizer:
    """Encode text using UTF-8 bytes and two sequence boundary tokens."""

    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, *, add_boundaries: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if add_boundaries:
            return [self.bos_id, *tokens, self.eos_id]
        return tokens

    def decode(self, tokens: list[int], *, errors: str = "strict") -> str:
        byte_tokens = (token for token in tokens if token not in (self.bos_id, self.eos_id))
        return bytes(byte_tokens).decode("utf-8", errors=errors)
