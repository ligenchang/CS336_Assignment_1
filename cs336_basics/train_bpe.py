import os
import collections
import regex as re
from typing import List, Tuple, Dict, Union, BinaryIO, Set, DefaultDict, Counter
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
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

def tokenize_chunk(filename: str, start: int, end: int, special_tokens: List[str]) -> collections.Counter:
    # Use a set for faster token lookup
    special_tokens_set = set(special_tokens)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Tokenizing chunk from {start} to {end} with special tokens: {special_tokens_set}")
    # Pre-encode special tokens for reuse
    special_tokens_encoded = {token: (token.encode("utf-8"),) for token in special_tokens_set}
    
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
            # Decode with possible leftover from previous buffer
            chunk = leftover + chunk_bytes.decode("utf-8", errors="ignore")
            # Try not to split multi-byte utf-8 chars: if not at end, may need to trim
            if read_end < chunk_end:
                # Find last complete character
                for i in range(1, 5):
                    try:
                        chunk_bytes[-i:].decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    i = 0
                if i > 0:
                    # Save incomplete bytes for next buffer
                    leftover = chunk[-i:]
                    chunk = chunk[:-i]
                else:
                    leftover = ""
            else:
                leftover = ""
            tokens = pre_tokenize_text_with_special(chunk, special_tokens)
            for token in tokens:
                if token in special_tokens_set:
                    word_freqs[special_tokens_encoded[token]] += 1
                else:
                    token_bytes = token.encode("utf-8")
                    word_freqs[tuple(bytes([b]) for b in token_bytes)] += 1
            pos = read_end
        mm.close()
    return word_freqs

def parallel_pretokenize(filename: str, special_tokens: List[str], num_workers: int) -> collections.Counter:
    # Optimize for smaller files by using fewer workers
    file_size = os.path.getsize(filename)
    effective_workers = min(num_workers, max(1, file_size // (1024 * 1024)))  # 1 worker per MB, at least 1
    
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Using {effective_workers} workers for pre-tokenization.")
    
    # Read file once to find boundaries
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, effective_workers, split_token)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Found {len(boundaries) - 1} chunk boundaries for parallel pre-tokenization.")
    
    # Create arguments for each chunk
    args = [
        (filename, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Starting parallel pre-tokenization...")
    
    # Process chunks in parallel
    word_freqs = collections.Counter()
    # for arg in args:
    #     word_freqs.update(tokenize_chunk(*arg))
    import concurrent
    # Cap workers to avoid OOM: max 2 for files >2GB, max 4 otherwise

    max_workers = effective_workers
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Using {max_workers} workers for parallel tokenization.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(tokenize_chunk, *arg) for arg in args]
        for future in concurrent.futures.as_completed(futures):
            word_freqs.update(future.result())

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Total tokens pre-tokenized: {sum(word_freqs.values())}")
    
    return word_freqs

class PairCounter:
    """
    Efficient pair counter that maintains pair frequencies and updates them incrementally.
    Similar to the approach used in tiktoken, this avoids recomputing all pair counts after each merge.
    """
    def __init__(self, word_freqs: Counter, skipped_pairs: Set[Tuple] = None):
        self.pair_freqs = collections.defaultdict(int)
        self.skipped_pairs = skipped_pairs or set()
        
        # Initialize pair counts directly - skip the word_pair_positions mapping for speed
        for word, freq in word_freqs.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs:
                    self.pair_freqs[pair] += freq
    
    def update_pairs(self, best_pair: Tuple, old_words: Dict[Tuple, int], new_words: Dict[Tuple, int]) -> None:
        """
        Fast update of pair frequencies after a merge.
        Only updates counts for pairs that could have changed.
        """
        bp0, bp1 = best_pair
        merged = bp0 + bp1
        
        # Decrement counts for pairs in old words
        for word, freq in old_words.items():
            if len(word) <= 1:
                continue
                
            # Find all adjacent pairs in the word and decrement their counts
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs and pair in self.pair_freqs:
                    self.pair_freqs[pair] -= freq
                    if self.pair_freqs[pair] <= 0:
                        del self.pair_freqs[pair]
        
        # Increment counts for pairs in new words
        for word, freq in new_words.items():
            if len(word) <= 1:
                continue
                
            # Find all adjacent pairs in the word and increment their counts
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs:
                    self.pair_freqs[pair] += freq
    
    def get_best_pair(self) -> Tuple[Tuple, int]:
        """Get the most frequent pair."""
        if not self.pair_freqs:
            return None, 0
            
        max_freq = max(self.pair_freqs.values())
        best_pairs = [p for p, freq in self.pair_freqs.items() if freq == max_freq]
        best_pair = max(best_pairs)
        return best_pair, max_freq
    
    def add_skipped_pair(self, pair: Tuple) -> None:
        """Add a pair to the skipped set and remove it from frequencies."""
        self.skipped_pairs.add(pair)
        if pair in self.pair_freqs:
            del self.pair_freqs[pair]

def count_pairs(word_freqs, skipped_pairs):
    """Legacy method maintained for compatibility."""
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            if pair not in skipped_pairs:
                pair_freqs[pair] += freq
    return pair_freqs

def apply_merge_chunk(chunk, best_pair, protected_words):
    bp0, bp1 = best_pair
    merged_token = bp0 + bp1
    new_word_freqs = collections.defaultdict(int)
    merged_count = 0

    for word, freq in chunk:
        if word in protected_words:
            new_word_freqs[word] += freq
            continue

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == bp0 and word[i + 1] == bp1:
                new_word.append(merged_token)
                i += 2
                merged_count += 1
            else:
                new_word.append(word[i])
                i += 1
        new_word_freqs[tuple(new_word)] += freq

    return new_word_freqs, merged_count

import time

def apply_merge_fast(word_freqs, best_pair, protected_words):
    bp0, bp1 = best_pair
    merged_token = bp0 + bp1
    new_word_freqs = {}  # Use regular dict instead of Counter for speed
    merged_count = 0
    # Only track words that were actually modified
    changed_words = {}
    
    # Fast lookup for protected words
    protected_set = set(protected_words)

    for word, freq in word_freqs.items():
        if word in protected_set:
            new_word_freqs[word] = freq
            continue

        # Check if this word contains the pair we're merging
        contains_pair = False
        for i in range(len(word) - 1):
            if word[i] == bp0 and word[i + 1] == bp1:
                contains_pair = True
                break
                
        if not contains_pair:
            # If the word doesn't contain the pair, keep it as is
            new_word_freqs[word] = freq
            continue
            
        # If we're here, the word contains the pair to merge
        changed_words[word] = freq
        
        # Perform the merge
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == bp0 and word[i + 1] == bp1:
                new_word.append(merged_token)
                i += 2
                merged_count += 1
            else:
                new_word.append(word[i])
                i += 1
        
        new_word_tuple = tuple(new_word)
        if new_word_tuple in new_word_freqs:
            new_word_freqs[new_word_tuple] += freq
        else:
            new_word_freqs[new_word_tuple] = freq

    return new_word_freqs, merged_count, changed_words

def train_bpe(
    input_path: Union[str, pathlib.Path],
    vocab_size: int,
    special_tokens: List[str] = [],
    num_workers: int = 8,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    input_path = str(input_path)
    timings = {}

    start = time.perf_counter()
    word_freqs = parallel_pretokenize(input_path, special_tokens, num_workers)
    timings["pretokenize"] = time.perf_counter() - start

    start = time.perf_counter()
    vocab = {i: bytes([i]) for i in range(256)}
    vocab_set = set(vocab.values())
    next_id = 256

    special_token_bytes = set(token.encode("utf-8") for token in special_tokens)
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab_set:
            vocab[next_id] = token_bytes
            vocab_set.add(token_bytes)
            next_id += 1
    timings["init_vocab"] = time.perf_counter() - start

    start = time.perf_counter()
    protected_words = set()
    for word in word_freqs:
        joined = b"".join(word)
        if any(st in joined for st in special_token_bytes):
            protected_words.add(word)
    timings["protect_special"] = time.perf_counter() - start

    skipped_pairs = set()
    merges = []

    # Initialize pair counter with word frequencies
    start = time.perf_counter()
    pair_counter = PairCounter(word_freqs, skipped_pairs)
    timings["init_pair_counter"] = time.perf_counter() - start

    merge_iterations = 0
    apply_merge_total_time = 0
    pair_update_total_time = 0
    get_best_pair_total_time = 0
    merge_loop_timing = {
        'get_best_pair': 0.0,
        'apply_merge': 0.0,
        'pair_update': 0.0
    }

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Starting training with vocab size {vocab_size}, initial vocab size {len(vocab)}")
    while len(vocab) < vocab_size:
        # Time get_best_pair
        t0 = time.perf_counter()
        best_pair, max_freq = pair_counter.get_best_pair()
        t1 = time.perf_counter()
        merge_loop_timing['get_best_pair'] += t1 - t0

        if not best_pair or max_freq == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Stopping: no more pairs to merge at iteration {merge_iterations}.")
            break

        if merge_iterations % 100 == 0 or len(vocab) >= vocab_size:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Iter {merge_iterations}: merging pair {best_pair} (freq={max_freq}), vocab size={len(vocab)}")

        new_token = best_pair[0] + best_pair[1]
        if new_token in vocab_set:
            pair_counter.add_skipped_pair(best_pair)
            continue

        vocab[next_id] = new_token
        vocab_set.add(new_token)
        merges.append(best_pair)
        next_id += 1

        # Time apply_merge_fast
        merge_start = time.perf_counter()
        word_freqs_dict, total_merged, changed_words = apply_merge_fast(word_freqs, best_pair, protected_words)
        merge_end = time.perf_counter()
        apply_merge_total_time += merge_end - merge_start
        merge_loop_timing['apply_merge'] += merge_end - merge_start

        if total_merged == 0:
            pair_counter.add_skipped_pair(best_pair)
            continue

        # Time update_pairs
        update_start = time.perf_counter()
        # Convert dict back to Counter if needed for compatibility
        word_freqs = collections.Counter(word_freqs_dict)
        # Get the affected new words (those containing the merged token)
        merged_token = best_pair[0] + best_pair[1]
        affected_new_words = {}
        for word, freq in word_freqs_dict.items():
            if any(tok == merged_token for tok in word):
                affected_new_words[word] = freq
        # Only update pair frequencies for words that changed
        pair_counter.update_pairs(best_pair, changed_words, affected_new_words)
        update_end = time.perf_counter()
        pair_update_total_time += update_end - update_start
        merge_loop_timing['pair_update'] += update_end - update_start

        merge_iterations += 1
        # if merge_iterations % 1000 == 0 or len(vocab) >= vocab_size:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Progress: {merge_iterations} merges, vocab size={len(vocab)}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Timing (per 1000 merges): get_best_pair={merge_loop_timing['get_best_pair']:.2f}s, apply_merge={merge_loop_timing['apply_merge']:.2f}s, pair_update={merge_loop_timing['pair_update']:.2f}s")
        merge_loop_timing = {'get_best_pair': 0.0, 'apply_merge': 0.0, 'pair_update': 0.0}

    total_time = sum(timings.values()) + apply_merge_total_time + pair_update_total_time

    print("\n===== Timing Report =====")
    for k, v in timings.items():
        print(f"{k:25s}: {v:.4f} sec")
    print(f"apply_merge (total)      : {apply_merge_total_time:.4f} sec")
    print(f"pair_update (total)      : {pair_update_total_time:.4f} sec")
    print(f"merge iterations         : {merge_iterations}")
    print(f"total                    : {total_time:.4f} sec")
    print("=========================")
    # Optionally print total time spent in each merge loop step
    print(f"get_best_pair (total)    : {merge_loop_timing['get_best_pair']:.4f} sec")
    print(f"apply_merge (total)      : {merge_loop_timing['apply_merge']:.4f} sec")
    print(f"pair_update (total)      : {merge_loop_timing['pair_update']:.4f} sec\n")

    return vocab, merges