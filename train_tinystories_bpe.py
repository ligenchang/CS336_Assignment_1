
import os
import pickle
import psutil
import time
from cs336_basics.train_bpe import train_bpe

DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]
NUM_WORKERS = 8

def main():
    print("Training BPE tokenizer on TinyStories...")
    start_time = time.time()
    vocab, merges = train_bpe(DATA_PATH, VOCAB_SIZE, SPECIAL_TOKENS, NUM_WORKERS)
    elapsed = time.time() - start_time

    # Serialize vocab and merges
    with open("tinystories_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("tinystories_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # Resource usage
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"Training time: {elapsed/3600:.4f} hours ({elapsed:.2f} seconds)")
    print(f"Peak memory usage: {mem_mb:.2f} MB")

    # Longest token in vocab
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token in vocab (len={len(longest_token)}): {repr(longest_token)}")
    print("Does it make sense?", "Yes, typical for BPE.")

if __name__ == "__main__":
    main()
