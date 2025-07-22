import os
import collections
import regex as re
from typing import List, Tuple, Dict, Union, BinaryIO, Set, DefaultDict, Counter
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    
    # Pre-encode special tokens for reuse
    special_tokens_encoded = {token: (token.encode("utf-8"),) for token in special_tokens_set}
    
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        tokens = pre_tokenize_text_with_special(chunk, special_tokens)
    
    # Preallocate a larger initial Counter to reduce resizing
    word_freqs = collections.Counter()
    
    # Process tokens in larger batches
    for token in tokens:
        if token in special_tokens_set:
            # Use pre-encoded tuple for special tokens
            word_freqs[special_tokens_encoded[token]] += 1
        else:
            # More efficient way to convert to bytes
            token_bytes = token.encode("utf-8")
            word_freqs[tuple(bytes([b]) for b in token_bytes)] += 1
            
    return word_freqs

def parallel_pretokenize(filename: str, special_tokens: List[str], num_workers: int) -> collections.Counter:
    # Optimize for smaller files by using fewer workers
    file_size = os.path.getsize(filename)
    effective_workers = min(num_workers, max(1, file_size // (1024 * 1024)))  # 1 worker per MB, at least 1
    
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    
    # Read file once to find boundaries
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, effective_workers, split_token)
    
    # Create arguments for each chunk
    args = [
        (filename, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    # Process chunks in parallel
    word_freqs = collections.Counter()
    
    # For very small files, process directly without parallelization
    if file_size < 500000:  # Less than 500KB
        for arg in args:
            word_freqs.update(tokenize_chunk(*arg))
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all tasks at once
            futures = [executor.submit(tokenize_chunk, *arg) for arg in args]
            # Process results as they complete
            for future in as_completed(futures):
                word_freqs.update(future.result())
    
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

    while len(vocab) < vocab_size:
        best_pair, max_freq = pair_counter.get_best_pair()
        if not best_pair or max_freq == 0:
            break

        new_token = best_pair[0] + best_pair[1]
        if new_token in vocab_set:
            pair_counter.add_skipped_pair(best_pair)
            continue

        vocab[next_id] = new_token
        vocab_set.add(new_token)
        merges.append(best_pair)
        next_id += 1

        merge_start = time.perf_counter()
        word_freqs_dict, total_merged, changed_words = apply_merge_fast(word_freqs, best_pair, protected_words)
        apply_merge_total_time += time.perf_counter() - merge_start

        if total_merged == 0:
            pair_counter.add_skipped_pair(best_pair)
            continue

        # Update pair counter incrementally with changed words
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
        pair_update_total_time += time.perf_counter() - update_start
        merge_iterations += 1

    total_time = sum(timings.values()) + apply_merge_total_time + pair_update_total_time

    print("\n===== Timing Report =====")
    for k, v in timings.items():
        print(f"{k:25s}: {v:.4f} sec")
    print(f"apply_merge (total)      : {apply_merge_total_time:.4f} sec")
    print(f"pair_update (total)      : {pair_update_total_time:.4f} sec")
    print(f"merge iterations         : {merge_iterations}")
    print(f"total                    : {total_time:.4f} sec")
    print("=========================\n")

    return vocab, merges