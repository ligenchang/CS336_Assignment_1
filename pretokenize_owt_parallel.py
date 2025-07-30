import os
import pickle
import multiprocessing
from cs336_basics.tokenizer import Tokenizer

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
    import time
    chunk_idx, start, end, raw_text_path, vocab, merges, special_token, tmp_dir = args
    from datetime import datetime
    def ts(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    t0 = time.time()
    ts(f"[Chunk {chunk_idx}] Starting tokenization (offsets {start}-{end})")
    ts(f"[Chunk {chunk_idx}] About to import sys...")
    import sys
    ts(f"[Chunk {chunk_idx}] About to import Tokenizer...")
    from cs336_basics.tokenizer import Tokenizer
    ts(f"[Chunk {chunk_idx}] Vocab and merges received from main process.")
    t_tok0 = time.time()
    ts(f"[Chunk {chunk_idx}] About to initialize Tokenizer...")
    tokenizer = Tokenizer(vocab, merges, special_tokens=[special_token])
    t_tok1 = time.time()
    ts(f"[Chunk {chunk_idx}] Tokenizer initialized in {t_tok1-t_tok0:.2f} sec.")
    ts(f"[Chunk {chunk_idx}] About to suppress stdout/stderr...")
    # sys.stdout = open(os.devnull, 'w')
    # sys.stderr = open(os.devnull, 'w')
    # print(f"[Chunk {chunk_idx}] Suppressed stdout/stderr. About to open file {raw_text_path}...")
    tokens = []
    chunk_size = 16 * 1024 * 1024  # 16MB sub-chunks
    t_mmap0 = time.time()
    ts(f"[Chunk {chunk_idx}] About to open raw text file {raw_text_path}...")
    with open(raw_text_path, "rb") as f:
        ts(f"[Chunk {chunk_idx}] File opened. About to import mmap...")
        import mmap
        ts(f"[Chunk {chunk_idx}] About to open mmap...")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        t_mmap1 = time.time()
        ts(f"[Chunk {chunk_idx}] mmap opened in {t_mmap1-t_mmap0:.2f} sec.")
        pos = start
        leftover = b""
        subchunk_idx = 0
        while pos < end:
            ts(f"[Chunk {chunk_idx}] Sub-chunk {subchunk_idx}: About to read and decode bytes {pos}-{min(pos+chunk_size, end)}...")
            read_end = min(pos + chunk_size, end)
            chunk_bytes = leftover + mm[pos:read_end]
            ts(f"[Chunk {chunk_idx}] Sub-chunk {subchunk_idx}: Read {len(chunk_bytes)} bytes.")
            # Try to decode as much as possible to utf-8
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
            ts(f"[Chunk {chunk_idx}] Sub-chunk {subchunk_idx}: {len(lines)} lines to tokenize.")
            tokens.extend(tokenizer.encode_iterable(lines))
            ts(f"[Chunk {chunk_idx}] Sub-chunk {subchunk_idx}: Tokenized. Total tokens so far: {len(tokens)}.")
            pos = read_end
            subchunk_idx += 1
        if leftover:
            try:
                ts(f"[Chunk {chunk_idx}] Final leftover: About to decode and tokenize.")
                chunk = leftover.decode("utf-8")
                lines = chunk.splitlines()
                ts(f"[Chunk {chunk_idx}] Final leftover: {len(lines)} lines to tokenize.")
                tokens.extend(tokenizer.encode_iterable(lines))
                ts(f"[Chunk {chunk_idx}] Final leftover: Tokenized. Total tokens so far: {len(tokens)}.")
            except Exception:
                ts(f"[Chunk {chunk_idx}] Exception decoding leftover bytes.")
                pass
        mm.close()
        ts(f"[Chunk {chunk_idx}] mmap closed.")
    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}.pkl")
    ts(f"[Chunk {chunk_idx}] About to write tokens to {tmp_path}...")
    with open(tmp_path, "wb") as f:
        pickle.dump(tokens, f)
    t1 = time.time()
    ts(f"[Chunk {chunk_idx}] Tokenized and saved {len(tokens)} tokens in {t1-t0:.2f} sec (offsets {start}-{end})")
    return tmp_path

def main():
    # File paths
    raw_text_path = "data/owt_valid.txt"
    vocab_path = "owt_bpe_vocab.pkl"
    merges_path = "owt_bpe_merges.pkl"
    output_tokens_path = "openwebtext_pretok_tokens.pkl"
    special_token = "<|endoftext|>"
    num_processes = 12

    import tempfile
    import time
    t_start = time.time()
    from datetime import datetime
    def ts(msg):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    ts("Finding chunk boundaries for parallel pretokenization...")
    t0 = time.time()
    with open(raw_text_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token.encode("utf-8"))
        chunk_offsets = [(i, start, end) for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
    t1 = time.time()
    ts(f"Found {len(chunk_offsets)} chunk boundaries in {t1-t0:.2f} sec.")

    # Create a temp directory for chunk token files
    # Load vocab and merges in main process
    ts("[Main] Loading vocab and merges in main process...")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    ts("[Main] Vocab loaded.")
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)
    ts("[Main] Merges loaded.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ts(f"Tokenizing {len(chunk_offsets)} chunks in parallel...")
        t2 = time.time()
        args = [
            (i, start, end, raw_text_path, vocab, merges, special_token, tmp_dir)
            for i, start, end in chunk_offsets
        ]
        ts("[Main] About to create multiprocessing pool...")
        try:
            with multiprocessing.Pool(num_processes) as pool:
                ts("[Main] Pool created, about to map chunks...")
                tmp_paths = pool.map(pretokenize_chunk, args)
                ts("[Main] Pool.map finished.")
        except Exception as e:
            print(f"[Main] Exception during pool.map: {e}")
            raise
        t3 = time.time()
        ts(f"All chunks tokenized in {t3-t2:.2f} sec.")

        # Aggregate all tokens from temp files, streaming to avoid OOM
        ts(f"Aggregating tokens from {len(tmp_paths)} chunk files...")
        t4 = time.time()
        total_tokens = 0
        with open(output_tokens_path, "wb") as out_f:
            for tmp_path in sorted(tmp_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
                ts(f"[Main] Aggregating {tmp_path}...")
                with open(tmp_path, "rb") as f:
                    tokens = pickle.load(f)
                    pickle.dump(tokens, out_f)
                    total_tokens += len(tokens)
        t5 = time.time()
        ts(f"Aggregated and saved {total_tokens} tokens in {t5-t4:.2f} sec.")
        ts(f"Saved pre-tokenized tokens to {output_tokens_path}.")
    t_end = time.time()
    ts(f"Total elapsed time: {t_end-t_start:.2f} sec.")
    ts("Done.")

if __name__ == "__main__":
    main()
