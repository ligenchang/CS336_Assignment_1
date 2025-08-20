import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer

tokens_path = "wikipedia_pretok_tokens.pkl"
vocab_path = "owt_bpe_vocab.pkl"  # Change if using a different vocab
merges_path = "owt_bpe_merges.pkl"  # Change if using a different merges
# Load vocab and merges from pickle files
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
with open(merges_path, "rb") as f:
    merges = pickle.load(f)
# Load tokenizer with loaded vocab and merges
tokenizer = Tokenizer(vocab, merges)


import random
import collections

# Collect a large sample of tokens for analysis
sample_size = 10000
all_tokens = []
with open(tokens_path, "rb") as f:
    while len(all_tokens) < sample_size:
        try:
            obj = pickle.load(f)
        except EOFError:
            break
        if isinstance(obj, (list, tuple, np.ndarray)):
            for t in obj:
                if len(all_tokens) < sample_size:
                    all_tokens.append(t)
                else:
                    break

print(f"[INFO] Collected {len(all_tokens)} tokens for analysis.")

# 1. Token frequency histogram
counter = collections.Counter(all_tokens)
most_common = counter.most_common(10)
print("[INFO] Top 10 most common token IDs:")
for tid, count in most_common:
    print(f"  Token {tid}: {count} times")

# 2. Unknown/special token count (assuming tokenizer.unk_token or id 0/1)
unk_token_id = getattr(tokenizer, 'unk_token_id', None)
if unk_token_id is None:
    unk_token_id = 0  # fallback, adjust if needed
unk_count = counter[unk_token_id]
print(f"[INFO] Unknown token (id={unk_token_id}) appears {unk_count} times in sample.")

# 3. Decode random samples
num_samples = 5
sample_len = 50
print("[INFO] Decoded random samples:")
for i in range(num_samples):
    start = random.randint(0, len(all_tokens) - sample_len)
    toks = all_tokens[start:start+sample_len]
    decoded = tokenizer.decode(toks)
    print(f"--- Sample {i+1} ---\n{decoded}\n")

# 4. Decode first 1000 tokens for reference
first_1000 = all_tokens[:1000]
decoded_text = tokenizer.decode(first_1000)
print("[INFO] Decoded text from first 1000 tokens:")
print(decoded_text)

# Count total number of tokens in the dataset
print("[INFO] Counting total number of tokens in the dataset...")
total_tokens = 0
with open(tokens_path, "rb") as f:
    while True:
        try:
            obj = pickle.load(f)
        except EOFError:
            break
        if isinstance(obj, (list, tuple, np.ndarray)):
            total_tokens += len(obj)
print(f"[INFO] Total number of tokens in {tokens_path}: {total_tokens:,}")
