import os
import collections
import regex as re
from typing import List, Tuple, Dict, Union, BinaryIO
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
    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        tokens = pre_tokenize_text_with_special(chunk, special_tokens)

    word_freqs = collections.Counter()
    for token in tokens:
        if token in special_tokens:
            word_freqs[(token.encode("utf-8"),)] += 1
        else:
            word_freqs[tuple(bytes([b]) for b in token.encode("utf-8"))] += 1
    return word_freqs

def parallel_pretokenize(filename: str, special_tokens: List[str], num_workers: int) -> collections.Counter:
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    args = [
        (filename, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    word_freqs = collections.Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(tokenize_chunk, *arg) for arg in args]
        for future in as_completed(futures):
            word_freqs.update(future.result())
    return word_freqs

def count_pairs(word_freqs, skipped_pairs):
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
    new_word_freqs = collections.Counter()
    merged_count = 0

    for word, freq in word_freqs.items():
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

    start = time.perf_counter()
    pair_freqs = count_pairs(word_freqs, skipped_pairs)
    timings["first_pair_count"] = time.perf_counter() - start

    merge_iterations = 0
    apply_merge_total_time = 0
    count_pairs_total_time = 0

    while len(vocab) < vocab_size:
        if not pair_freqs:
            break

        max_freq = max(pair_freqs.values())
        best_pairs = [p for p, freq in pair_freqs.items() if freq == max_freq]
        best_pair = max(best_pairs)

        new_token = best_pair[0] + best_pair[1]
        if new_token in vocab_set:
            skipped_pairs.add(best_pair)
            pair_freqs.pop(best_pair, None)
            continue

        vocab[next_id] = new_token
        vocab_set.add(new_token)
        merges.append(best_pair)
        next_id += 1

        merge_start = time.perf_counter()
        word_freqs, total_merged = apply_merge_fast(word_freqs, best_pair, protected_words)
        apply_merge_total_time += time.perf_counter() - merge_start

        if total_merged == 0:
            skipped_pairs.add(best_pair)
            pair_freqs.pop(best_pair, None)
            continue

        count_start = time.perf_counter()
        pair_freqs = count_pairs(word_freqs, skipped_pairs)
        count_pairs_total_time += time.perf_counter() - count_start

        merge_iterations += 1

    total_time = sum(timings.values()) + apply_merge_total_time + count_pairs_total_time

    print("\n===== Timing Report =====")
    for k, v in timings.items():
        print(f"{k:25s}: {v:.4f} sec")
    print(f"apply_merge (total)      : {apply_merge_total_time:.4f} sec")
    print(f"count_pairs (total)      : {count_pairs_total_time:.4f} sec")
    print(f"merge iterations         : {merge_iterations}")
    print(f"total                    : {total_time:.4f} sec")
    print("=========================\n")

    return vocab, merges