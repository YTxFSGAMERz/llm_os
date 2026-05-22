import tiktoken

class Tokenizer:
    def __init__(self, encoding_name="gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        # We might need to add special tokens later, but for Phase 1A standard is fine.

    @property
    def vocab_size(self):
        return self.enc.n_vocab

    def encode(self, text, allowed_special="all"):
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens):
        return self.enc.decode(tokens)
