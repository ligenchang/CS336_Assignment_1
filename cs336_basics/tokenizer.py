"""
Implementation of a byte-level BPE tokenizer.
"""
import regex as re
from typing import Dict, List, Tuple, Set, Iterator, Union, Optional, Any
import collections
import itertools

class Tokenizer:
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        """
        Construct a Tokenizer from pickled vocab and merges files, and optional special tokens.
        vocab_filepath: path to vocab file (expects pickle: {id: bytes})
        merges_filepath: path to merges file (expects pickle: list of (bytes, bytes))
        special_tokens: list of special tokens (str)
        """
        import pickle
        # Load vocab
        with open(vocab_filepath, 'rb') as vf:
            vocab = pickle.load(vf)
        # Load merges
        with open(merges_filepath, 'rb') as mf:
            merges = pickle.load(mf)

        # Add special tokens to vocab if not present
        if special_tokens:
            max_id = max(vocab.keys(), default=-1)
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes not in vocab.values():
                    max_id += 1
                    vocab[max_id] = token_bytes

        return cls(vocab, merges, special_tokens)
    """
    A byte-level BPE tokenizer.
    """
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Initialize the tokenizer with a vocabulary, merges, and special tokens.
        
        Args:
            vocab: A dictionary mapping token ids to byte sequences
            merges: A list of tuples, each containing a pair of byte sequences to merge
            special_tokens: Optional list of special tokens that should never be split
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Create a mapping from byte sequences to token ids
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # Create dictionary of merge pairs
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        
        # Convert special tokens to bytes
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]
        
        # Sort special tokens by length (longest first) to handle overlapping tokens correctly
        self.special_tokens_bytes_sorted = sorted(
            [(token, i) for i, token in enumerate(self.special_tokens_bytes)],
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        # Compile a regex pattern to find special tokens
        if self.special_tokens:
            # Escape special regex characters in the tokens
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            # Sort by length (longest first) to handle overlapping tokens
            escaped_tokens.sort(key=len, reverse=True)
            # Create a pattern that matches any of the special tokens
            self.special_token_pattern = re.compile(
                '|'.join(f'({token})' for token in escaped_tokens),
                re.DOTALL
            )
        else:
            self.special_token_pattern = None

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into a list of token ids.
        
        Args:
            text: The string to encode
            
        Returns:
            A list of token ids
        """
        byte_encoded = text.encode('utf-8')
        tokens = []
        
        # Check for special tokens first
        if self.special_tokens and byte_encoded:
            tokens = self._encode_with_special_tokens(text)
        else:
            # If no special tokens or empty string, just tokenize normally
            tokens = self._bpe_encode(byte_encoded)
            
        return tokens

    def _encode_with_special_tokens(self, text: str) -> List[int]:
        """
        Encode a string with special token handling.
        
        Args:
            text: The string to encode
            
        Returns:
            A list of token ids
        """
        tokens = []
        
        if self.special_token_pattern:
            special_matches = list(self.special_token_pattern.finditer(text))
            
            if not special_matches:
                return self._bpe_encode(text.encode('utf-8'))
            
            last_end = 0
            for match in special_matches:
                start, end = match.span()
                
                if start > last_end:
                    before_text = text[last_end:start]
                    tokens.extend(self._bpe_encode(before_text.encode('utf-8')))
                
                special_token = match.group(0)
                if special_token in self.special_tokens:
                    special_token_bytes = special_token.encode('utf-8')
                    if special_token_bytes in self.byte_to_id:
                        tokens.append(self.byte_to_id[special_token_bytes])
                    else:
                        tokens.extend(self._bpe_encode(special_token_bytes))
                
                # Check if consecutive newlines follow this special token
                next_pos = end
                newline_count = 0
                while next_pos < len(text) and text[next_pos] == '\n':
                    newline_count += 1
                    next_pos += 1
                
                # Check if there's non-whitespace content after the newlines
                has_content_after = False
                if next_pos < len(text):
                    remaining = text[next_pos:]
                    has_content_after = bool(remaining.strip())
                
                # Handle differently based on whether there's content after newlines
                if newline_count >= 2:
                    if has_content_after:
                        # For newlines followed by non-whitespace content, keep separate
                        newline_byte = b'\n'
                        for _ in range(newline_count):
                            tokens.append(self.byte_to_id[newline_byte])
                    else:
                        # For trailing newlines only, merge them
                        # Use the merged token for "\n\n"
                        merged_newlines = b'\n' * newline_count
                        if merged_newlines in self.byte_to_id:
                            tokens.append(self.byte_to_id[merged_newlines])
                        else:
                            # Fallback if specific merged token isn't in vocab
                            newline_byte = b'\n'
                            tokens.append(self.byte_to_id[newline_byte])
                            if newline_count > 1:
                                tokens.append(self.byte_to_id[newline_byte])
                    
                    last_end = next_pos
                else:
                    last_end = end
            
            if last_end < len(text):
                tokens.extend(self._bpe_encode(text[last_end:].encode('utf-8')))
                    
        return tokens


    def _bpe_encode(self, byte_encoded: bytes) -> List[int]:
        """
        Encode a byte string using BPE.

        Args:
            byte_encoded: The UTF-8 encoded bytes

        Returns:
            A list of token ids
        """
        if not byte_encoded:
            return []

        tokens = [bytes([b]) for b in byte_encoded]
        merge_ranks = self.merge_ranks
        byte_to_id = self.byte_to_id

        while len(tokens) > 1:
            best_pair = None
            best_rank = float('inf')

            # Identify best mergeable pair in a single pass
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in merge_ranks:
                    rank = merge_ranks[pair]
                    merged = pair[0] + pair[1]
                    if merged in byte_to_id and rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            # Merge all instances of the best pair
            result = []
            i = 0
            first, second = best_pair
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    result.append(first + second)
                    i += 2
                else:
                    result.append(tokens[i])
                    i += 1

            tokens = result

        return [byte_to_id[tok] for tok in tokens if tok in byte_to_id]

        
        # Convert tokens to ids efficiently
        result = []
        for token in tokens:
            # Most common case: token is directly in the vocabulary
            if token in byte_to_id:
                result.append(byte_to_id[token])
            else:
                # If a token isn't in our vocabulary, split it into individual bytes
                for b in token:
                    byte_token = bytes([b])
                    if byte_token in byte_to_id:
                        result.append(byte_to_id[byte_token])
        
        return result

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token ids back to a string.
        
        Args:
            token_ids: List of token ids
            
        Returns:
            The decoded string
        """
        # Convert token ids back to byte sequences
        byte_sequences = [self.vocab[token_id] for token_id in token_ids if token_id in self.vocab]
        
        # Concatenate all byte sequences and decode to string
        if not byte_sequences:
            return ""
        
        return b''.join(byte_sequences).decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        Logs progress every 10000 lines.
        """
        count = 0
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id
            count += 1
            if count % 10000 == 0:
                print(f"[Tokenizer.encode_iterable] Processed {count} lines...")
