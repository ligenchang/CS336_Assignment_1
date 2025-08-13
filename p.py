
import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer

tokens_path = "openwebtext_pretok_tokens.pkl"
vocab_path = "owt_bpe_vocab.pkl"  # Change if using a different vocab
merges_path = "owt_bpe_merges.pkl"  # Change if using a different merges
# Load vocab and merges from pickle files
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
with open(merges_path, "rb") as f:
    merges = pickle.load(f)
# Load tokenizer with loaded vocab and merges
tokenizer = Tokenizer(vocab, merges)

# Collect first 1000 tokens
first_1000 = []
with open(tokens_path, "rb") as f:
    while len(first_1000) < 1000:
        try:
            obj = pickle.load(f)
        except EOFError:
            break
        if isinstance(obj, (list, tuple, np.ndarray)):
            for t in obj:
                if len(first_1000) < 1000:
                    first_1000.append(t)
                else:
                    break

print(f"[INFO] Collected {len(first_1000)} tokens.")

# Map tokens to words
decoded_text = tokenizer.decode(first_1000)
print("[INFO] Decoded text from first 1000 tokens:")
print(decoded_text)
