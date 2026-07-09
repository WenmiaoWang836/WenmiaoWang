# tokenizer.py
class EnhancedTokenizer:
    def __init__(self, max_len=32):
        self.max_len = max_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
    def build_vocab(self, texts):
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        for char, _ in sorted_chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
    def encode(self, text):
        ids = [self.vocab.get(c, 1) for c in text[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return ids