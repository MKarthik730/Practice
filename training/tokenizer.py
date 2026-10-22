class ByteTokenizer:
    """UTF-8 byte tokenizer with fixed IDs and no external vocabulary files."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    byte_offset = 4
    vocab_size = 260

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        raw = bytearray()
        special_tokens = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        for token_id in ids:
            if token_id in special_tokens:
                if not skip_special_tokens:
                    raw.extend(f"<{token_id}>".encode("ascii"))
                continue
            if not self.byte_offset <= token_id < self.vocab_size:
                raise ValueError(f"token id out of range: {token_id}")
            raw.append(token_id - self.byte_offset)
        return raw.decode("utf-8", errors="replace")
