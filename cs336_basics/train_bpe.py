import os
import collections
import regex as re
from typing import List, Tuple, Dict, Union, Set
import pathlib
import mmap
import time
from datetime import datetime

# GPT-2 style pre-tokenizer regex
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def split_text_by_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    escaped_tokens = [re.escape(tok) for tok in special_tokens]
    pattern = "(" + "|".join(escaped_tokens) + ")"
    parts = re.split(pattern, text)
    return [p for p in parts if p]

def pre_tokenize_text_with_special(text: str, special_tokens: List[str]) -> List[str]:
    parts = split_text_by_special_tokens(text, special_tokens)
    tokens = []
    for part in parts:
        if part in special_tokens:
            tokens.append(part)
        else:
            tokens.extend([m.group() for m in GPT2_PATTERN.finditer(part)])
    return tokens

def find_chunk_boundaries(file: mmap.mmap, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
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

def tokenize_chunk(filename: str, start: int, end: int, special_tokens: List[str], token_to_id: Dict[bytes, int]) -> collections.Counter:
    special_tokens_set = set(special_tokens)
    special_tokens_encoded = {token: token.encode("utf-8") for token in special_tokens_set}

    import mmap
    buffer_size = 64 * 1024 * 1024  # 64MB sub-buffers
    word_freqs = collections.Counter()
    with open(filename, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = mm.size()
        chunk_end = min(end, file_size)
        pos = start
        leftover = ""
        while pos < chunk_end:
            read_end = min(pos + buffer_size, chunk_end)
            chunk_bytes = mm[pos:read_end]
            chunk = leftover + chunk_bytes.decode("utf-8", errors="ignore")
            if read_end < chunk_end:
                for i in range(1, 5):
                    try:
                        chunk_bytes[-i:].decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    i = 0
                if i > 0:
                    leftover = chunk[-i:]
                    chunk = chunk[:-i]
                else:
                    leftover = ""
            else:
                leftover = ""

            tokens = pre_tokenize_text_with_special(chunk, special_tokens)
            for token in tokens:
                if token in special_tokens_set:
                    # Special token: encode and map to single token ID
                    token_bytes = special_tokens_encoded[token]
                    token_id = token_to_id.setdefault(token_bytes, len(token_to_id))
                    word_freqs[(token_id,)] += 1
                else:
                    token_bytes = token.encode("utf-8")
                    # Convert bytes to token IDs (one per byte)
                    token_ids = tuple(token_to_id.setdefault(bytes([b]), len(token_to_id)) for b in token_bytes)
                    word_freqs[token_ids] += 1
            pos = read_end
        mm.close()
    return word_freqs

def parallel_pretokenize(filename: str, special_tokens: List[str], num_workers: int, token_to_id: Dict[bytes, int]) -> collections.Counter:
    file_size = os.path.getsize(filename)
    effective_workers = min(num_workers, max(1, file_size // (1024 * 1024)))
    
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, effective_workers, split_token)

    args = [
        (filename, start, end, special_tokens, token_to_id)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    word_freqs = collections.Counter()
    import concurrent.futures
    max_workers = effective_workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(tokenize_chunk, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            word_freqs.update(future.result())

    return word_freqs

class PairCounter:
    def __init__(self, word_freqs: collections.Counter, skipped_pairs: Set[Tuple[int, int]] = None, id_to_token: Dict[int, bytes] = None):
        self.pair_freqs = collections.defaultdict(int)
        self.skipped_pairs = skipped_pairs or set()
        self.id_to_token = id_to_token

        for word, freq in word_freqs.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs:
                    self.pair_freqs[pair] += freq
    
    def update_pairs(self, best_pair: Tuple[int, int], old_words: Dict[Tuple[int, ...], int], new_words: Dict[Tuple[int, ...], int]) -> None:
        bp0, bp1 = best_pair
        merged = bp0 * 256 + bp1  # This is just an example; actually, we represent merged tokens as tuple or int IDs elsewhere

        # Decrement old pairs
        for word, freq in old_words.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs and pair in self.pair_freqs:
                    self.pair_freqs[pair] -= freq
                    if self.pair_freqs[pair] <= 0:
                        del self.pair_freqs[pair]
        # Increment new pairs
        for word, freq in new_words.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs:
                    self.pair_freqs[pair] += freq
    
    def get_best_pair(self) -> Tuple[Tuple[int, int], int]:
        if not self.pair_freqs:
            return None, 0
        max_freq = max(self.pair_freqs.values())
        best_pairs = [p for p, freq in self.pair_freqs.items() if freq == max_freq]
        # Break ties by lexicographic order of the pair as bytes
        if self.id_to_token is not None:
            best_pair = max(best_pairs, key=lambda p: (self.id_to_token[p[0]], self.id_to_token[p[1]]))
        else:
            best_pair = max(best_pairs)
        return best_pair, max_freq
    
    def add_skipped_pair(self, pair: Tuple[int, int]) -> None:
        self.skipped_pairs.add(pair)
        if pair in self.pair_freqs:
            del self.pair_freqs[pair]

def apply_merge_fast(word_freqs: Dict[Tuple[int, ...], int], best_pair: Tuple[int, int], protected_words: Set[Tuple[int, ...]], merged_token_id: int):
    bp0, bp1 = best_pair
    new_word_freqs = {}
    merged_count = 0
    changed_words = {}

    for word, freq in word_freqs.items():
        if word in protected_words:
            new_word_freqs[word] = freq
            continue

        contains_pair = False
        for i in range(len(word) - 1):
            if word[i] == bp0 and word[i + 1] == bp1:
                contains_pair = True
                break

        if not contains_pair:
            new_word_freqs[word] = freq
            continue

        changed_words[word] = freq

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == bp0 and word[i + 1] == bp1:
                new_word.append(merged_token_id)
                i += 2
                merged_count += 1
            else:
                new_word.append(word[i])
                i += 1

        new_word_tuple = tuple(new_word)
        new_word_freqs[new_word_tuple] = new_word_freqs.get(new_word_tuple, 0) + freq

    return new_word_freqs, merged_count, changed_words

def train_bpe(
    input_path: Union[str, pathlib.Path],
    vocab_size: int,
    special_tokens: List[str] = [],
    num_workers: int = 8,
) -> Tuple[Dict[int, bytes], List[Tuple[int, int]]]:
    input_path = str(input_path)
    timings = {}

    # Initialize token_to_id and id_to_token for all byte tokens (0-255)
    token_to_id = {bytes([i]): i for i in range(256)}
    id_to_token = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # Add special tokens to token_to_id and id_to_token
    for st in special_tokens:
        b_st = st.encode("utf-8")
        if b_st not in token_to_id:
            token_to_id[b_st] = next_id
            id_to_token[next_id] = b_st
            next_id += 1

    start = time.perf_counter()
    # Pre-tokenize and get frequencies as int-token tuples
    word_freqs = parallel_pretokenize(input_path, special_tokens, num_workers, token_to_id)
    timings["pretokenize"] = time.perf_counter() - start

    # Protect words containing special tokens (by ID tuple)
    special_token_ids = set(token_to_id[st.encode("utf-8")] for st in special_tokens)
    protected_words = set()
    for word in word_freqs:
        if any(tok in special_token_ids for tok in word):
            protected_words.add(word)
    timings["protect_special"] = time.perf_counter() - start

    skipped_pairs = set()
    merges: List[Tuple[bytes, bytes]] = []

    start = time.perf_counter()
    pair_counter = PairCounter(word_freqs, skipped_pairs, id_to_token)
    timings["init_pair_counter"] = time.perf_counter() - start

    merge_iterations = 0
    apply_merge_total_time = 0
    pair_update_total_time = 0
    merge_loop_timing = {
        'get_best_pair': 0.0,
        'apply_merge': 0.0,
        'pair_update': 0.0
    }

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Starting training with vocab size {vocab_size}, initial vocab size {len(token_to_id)}")
    while len(token_to_id) < vocab_size:
        t0 = time.perf_counter()
        best_pair, max_freq = pair_counter.get_best_pair()
        t1 = time.perf_counter()
        merge_loop_timing['get_best_pair'] += t1 - t0

        if not best_pair or max_freq == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] No more pairs to merge at iteration {merge_iterations}.")
            break

        if merge_iterations % 100 == 0 or len(token_to_id) >= vocab_size:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Iter {merge_iterations}: merging pair {best_pair} (freq={max_freq}), vocab size={len(token_to_id)}")

        # Add new merged token to vocab
        merged_bytes = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        if merged_bytes in token_to_id:
            # Already merged before, skip
            pair_counter.add_skipped_pair(best_pair)
            continue
        token_to_id[merged_bytes] = next_id
        id_to_token[next_id] = merged_bytes
        merges.append((id_to_token[best_pair[0]], id_to_token[best_pair[1]]))
        merged_token_id = next_id
        next_id += 1

        # Apply merge fast: replace pairs in word_freqs with merged token ID
        merge_start = time.perf_counter()
        # Note: apply_merge_fast expects token IDs, merged token is new ID (int)
        word_freqs_dict, total_merged, changed_words = apply_merge_fast(word_freqs, best_pair, protected_words, merged_token_id)
        merge_end = time.perf_counter()
        apply_merge_total_time += merge_end - merge_start
        merge_loop_timing['apply_merge'] += merge_end - merge_start

        if total_merged == 0:
            pair_counter.add_skipped_pair(best_pair)
            continue

        # Update pairs
        update_start = time.perf_counter()
        word_freqs = collections.Counter(word_freqs_dict)

        # Filter new words containing the merged token ID
        affected_new_words = {w: f for w, f in word_freqs_dict.items() if merged_token_id in w}

        pair_counter.update_pairs(best_pair, changed_words, affected_new_words)
        update_end = time.perf_counter()
        pair_update_total_time += update_end - update_start
        merge_loop_timing['pair_update'] += update_end - update_start

        merge_iterations += 1

    total_time = sum(timings.values()) + apply_merge_total_time + pair_update_total_time

    print("\n===== Timing Report =====")
    for k, v in timings.items():
        print(f"{k:25s}: {v:.4f} sec")
    print(f"apply_merge (total)      : {apply_merge_total_time:.4f} sec")
    print(f"pair_update (total)      : {pair_update_total_time:.4f} sec")
    print(f"merge iterations         : {merge_iterations}")
    print(f"total                    : {total_time:.4f} sec")
    print("=========================")
    print(f"get_best_pair (total)    : {merge_loop_timing['get_best_pair']:.4f} sec")
    print(f"apply_merge (total)      : {merge_loop_timing['apply_merge']:.4f} sec")
    print(f"pair_update (total)      : {merge_loop_timing['pair_update']:.4f} sec\n")

    return id_to_token, merges
