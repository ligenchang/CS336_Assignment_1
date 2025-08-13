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

    chunk_size = 64 * 1024 * 1024  # read size
    batch_size = 5000               # number of lines per batch before writing

    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}.pkl")
    with open(raw_text_path, "rb") as f, open(tmp_path, "wb") as out_f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        pos = start
        leftover = b""
        total_bytes = end - start
        processed_bytes = 0

        lines_buffer = []

        while pos < end:
            read_end = min(pos + chunk_size, end)
            chunk_bytes = leftover + mm[pos:read_end]

            # Safe decode
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

            lines_buffer.extend(chunk.splitlines())

            # Process in batches
            while len(lines_buffer) >= batch_size:
                batch = lines_buffer[:batch_size]
                del lines_buffer[:batch_size]
                tokens = tokenizer.encode_iterable(batch)
                pickle.dump(list(tokens), out_f)

                # Update processed bytes based on estimated size of this batch
                batch_bytes = sum(len(line.encode("utf-8")) for line in batch)
                processed_bytes += batch_bytes
                percent = 100.0 * processed_bytes / total_bytes
                ts(f"[Chunk {chunk_idx}] Progress: {percent:.1f}% ({processed_bytes}/{total_bytes} bytes)")

            pos = read_end

        # Remaining lines
        if lines_buffer:
            tokens = tokenizer.encode_iterable(lines_buffer)
            pickle.dump(list(tokens), out_f)
            batch_bytes = sum(len(line.encode("utf-8")) for line in lines_buffer)
            processed_bytes += batch_bytes
            percent = 100.0 * processed_bytes / total_bytes
            ts(f"[Chunk {chunk_idx}] Progress: {percent:.1f}% ({processed_bytes}/{total_bytes} bytes)")

        # Leftover
        if leftover:
            try:
                chunk = leftover.decode("utf-8")
                tokens = tokenizer.encode_iterable(chunk.splitlines())
                pickle.dump(list(tokens), out_f)
                batch_bytes = len(leftover)
                processed_bytes += batch_bytes
                percent = 100.0 * processed_bytes / total_bytes
                ts(f"[Chunk {chunk_idx}] Progress: {percent:.1f}% ({processed_bytes}/{total_bytes} bytes)")
            except Exception:
                pass

        mm.close()

    t1 = time.time()
    ts(f"[Chunk {chunk_idx}] Tokenized directly to disk in {t1 - t0:.2f} sec")
    return tmp_path



def main():
    raw_text_path = "data/owt_train.txt"
    # vocab_path = "owt_bpe_vocab.pkl" 
    # merges_path = "owt_bpe_merges.pkl" 
    vocab_path = "owt_bpe_vocab.pkl" 
    merges_path = "owt_bpe_merges.pkl" 
    output_tokens_path = "openwebtext_pretok_tokens.pkl"
    special_token = "<|endoftext|>"
    num_processes = 12

    # Clean up any existing output file before starting
    if os.path.exists(output_tokens_path):
        ts(f"Removing existing output file: {output_tokens_path}")
        os.remove(output_tokens_path)

    # Clean up any existing temp files from previous runs
    import glob
    import tempfile as tf
    
    # Check current directory
    temp_files = glob.glob("chunk_*.pkl")
    if temp_files:
        ts(f"Removing {len(temp_files)} existing temp files from current directory")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except FileNotFoundError:
                pass
    
    # Check system temp directory for leftover chunk files
    temp_dir = tf.gettempdir()
    temp_files_in_tmp = glob.glob(os.path.join(temp_dir, "**/chunk_*.pkl"), recursive=True)
    if temp_files_in_tmp:
        ts(f"Removing {len(temp_files_in_tmp)} existing temp files from system temp directory")
        for temp_file in temp_files_in_tmp:
            try:
                os.remove(temp_file)
            except (FileNotFoundError, PermissionError):
                pass

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
        # with open(output_tokens_path, "wb") as out_f:
        #     for tmp_path in sorted(tmp_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
        #         with open(tmp_path, "rb") as f:
        #             tokens = pickle.load(f)
        #             pickle.dump(tokens, out_f)
        #             total_tokens += len(tokens)

        with open(output_tokens_path, "wb") as out_f:
            for tmp_path in sorted(tmp_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
                with open(tmp_path, "rb") as f:
                    try:
                        while True:
                            token_batch = pickle.load(f)
                            pickle.dump(token_batch, out_f)
                            total_tokens += len(token_batch)
                    except EOFError:
                        pass

        ts(f"Saved {total_tokens} tokens to {output_tokens_path}.")

    ts(f"Done in {time.time() - t_start:.2f} sec.")


if __name__ == "__main__":
    main()
