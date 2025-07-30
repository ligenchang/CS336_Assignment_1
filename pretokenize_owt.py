import os
import pickle
from cs336_basics.tokenizer import Tokenizer

def main():
    # Adjust these paths as needed
    raw_text_path = "data/owt_train.txt"  # Path to raw OWT text
    vocab_path = "owt_bpe_vocab.pkl"       # Path to BPE vocab
    merges_path = "owt_bpe_merges.pkl"     # Path to BPE merges
    output_tokens_path = "openwebtext_pretok_tokens.pkl"  # Output path

    print(f"Loading BPE vocab from {vocab_path} and merges from {merges_path}...")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges)

    print(f"Tokenizing raw text from {raw_text_path} in chunks using mmap...")
    import mmap
    tokens = []
    chunk_size = 16 * 1024 * 1024  # 16MB chunks
    with open(raw_text_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = mm.size()
        pos = 0
        leftover = b""
        while pos < file_size:
            read_end = min(pos + chunk_size, file_size)
            chunk_bytes = leftover + mm[pos:read_end]
            # Try to decode as much as possible to utf-8
            for i in range(4, -1, -1):
                try:
                    chunk = chunk_bytes[:len(chunk_bytes)-i].decode("utf-8")
                    leftover = chunk_bytes[len(chunk_bytes)-i:] if i > 0 else b""
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # If all fail, skip the last byte
                chunk = chunk_bytes[:-1].decode("utf-8", errors="ignore")
                leftover = chunk_bytes[-1:]
            tokens.extend(tokenizer.encode(chunk))
            pos = read_end
        if leftover:
            try:
                chunk = leftover.decode("utf-8")
                tokens.extend(tokenizer.encode(chunk))
            except Exception:
                pass
        mm.close()
    print(f"Saving tokenized data to {output_tokens_path}...")
    with open(output_tokens_path, "wb") as f:
        pickle.dump(tokens, f)
    print("Done.")

if __name__ == "__main__":
    main()
