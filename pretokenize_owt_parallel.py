import os
import pickle
import multiprocessing
from datetime import datetime
import time
import mmap
import tempfile
from cs336_basics.tokenizer import Tokenizer


def ts(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")




def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(args):

    chunk_idx, start, end, raw_text_path, tmp_dir, vocab_path, merges_path, special_token = args

    t0 = time.time()
    ts(f"[Chunk {chunk_idx}] Starting tokenization (offsets {start}-{end})")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[special_token])

    tokens = []
    chunk_size = 64 * 1024 * 1024  # 64MB

    with open(raw_text_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        pos = start
        leftover = b""
        total_bytes = end - start
        subchunk_idx = 0

        while pos < end:
            read_end = min(pos + chunk_size, end)
            chunk_bytes = leftover + mm[pos:read_end]

            # Decode carefully
            for i in range(4, -1, -1):
                try:
                    chunk = chunk_bytes[:len(chunk_bytes)-i].decode("utf-8")
                    leftover = chunk_bytes[len(chunk_bytes)-i:] if i > 0 else b""
                    break
                except UnicodeDecodeError:
                    continue
            else:
                chunk = chunk_bytes[:-1].decode("utf-8", errors="ignore")
                leftover = chunk_bytes[-1:]

            lines = chunk.splitlines()
            tokens.extend(tokenizer.encode_iterable(lines))

            pos = read_end
            subchunk_idx += 1

        if leftover:
            try:
                chunk = leftover.decode("utf-8")
                lines = chunk.splitlines()
                tokens.extend(tokenizer.encode_iterable(lines))
            except Exception:
                pass
        mm.close()

    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(tokens, f)

    t1 = time.time()
    ts(f"[Chunk {chunk_idx}] Tokenized and saved {len(tokens)} tokens in {t1 - t0:.2f} sec")
    return tmp_path


def main():
    raw_text_path = "data/owt_train.txt"
    vocab_path = "owt_bpe_vocab.pkl" 
    merges_path = "owt_bpe_merges.pkl" 
    output_tokens_path = "openwebtext_pretok_tokens.pkl"
    special_token = "<|endoftext|>"
    num_processes = 12

    t_start = time.time()
    ts("Finding chunk boundaries...")
    with open(raw_text_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token.encode("utf-8"))
    chunk_offsets = [(i, start, end) for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
    ts(f"Found {len(chunk_offsets)} chunks.")

    ts("Loading tokenizer from files...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[special_token])
    ts("Loaded tokenizer.")


    with tempfile.TemporaryDirectory() as tmp_dir:
        ts("Starting multiprocessing pool...")
        args = [
            (i, start, end, raw_text_path, tmp_dir, vocab_path, merges_path, special_token)
            for i, start, end in chunk_offsets
        ]
        with multiprocessing.Pool(num_processes) as pool:
            tmp_paths = pool.map(pretokenize_chunk, args)

        ts("Tokenization complete. Aggregating tokens...")
        total_tokens = 0
        with open(output_tokens_path, "wb") as out_f:
            for tmp_path in sorted(tmp_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
                with open(tmp_path, "rb") as f:
                    tokens = pickle.load(f)
                    pickle.dump(tokens, out_f)
                    total_tokens += len(tokens)

        ts(f"Saved {total_tokens} tokens to {output_tokens_path}.")

    ts(f"Done in {time.time() - t_start:.2f} sec.")


if __name__ == "__main__":
    main()
