import os
import sys
import time
import json
import argparse
import resource
from pathlib import Path

# Import the BPE training function from cs336_basics
from cs336_basics.train_bpe import train_bpe

def convert_bytes_to_unicode_for_display(token_bytes):
    """Convert token bytes to a readable unicode string for display purposes"""
    try:
        return token_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return repr(token_bytes)

def get_memory_usage():
    """Get current memory usage in GB"""
    # Use resource module which is standard in Unix-like systems (including macOS)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # maxrss is in KB on macOS, convert to GB
    return usage.ru_maxrss / (1024 * 1024)  # Convert to GB

def save_merges(merges, output_path):
    """Save merges to a text file in the format expected by tokenizers"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for first, second in merges:
            # Convert bytes to their string representation for saving
            try:
                first_str = first.decode('utf-8', errors='replace')
                second_str = second.decode('utf-8', errors='replace')
                f.write(f"{first_str} {second_str}\n")
            except:
                # For bytes that can't be decoded, use their hex representation
                first_str = ' '.join([f"0x{b:02x}" for b in first])
                second_str = ' '.join([f"0x{b:02x}" for b in second])
                f.write(f"{first_str} {second_str}\n")

def save_vocab(vocab, output_path):
    """Save vocabulary to a JSON file"""
    # Convert bytes to strings for JSON serialization
    vocab_str = {}
    for token_id, token_bytes in vocab.items():
        # Try to decode as UTF-8, fallback to hex representation
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
        except:
            token_str = ' '.join([f"0x{b:02x}" for b in token_bytes])
        vocab_str[token_id] = token_str
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_str, f, indent=2, ensure_ascii=False)

def find_longest_token(vocab):
    """Find the longest token in the vocabulary"""
    if not vocab:
        return None, 0
    
    longest_token = max(vocab.values(), key=len)
    return longest_token, len(longest_token)

def main(args):
    # Start timing
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train BPE tokenizer
    print(f"Training BPE tokenizer on {args.input_path} with vocab size {args.vocab_size}...")
    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        num_workers=args.num_workers
    )
    
    # Calculate time and memory usage
    end_time = time.time()
    final_memory = get_memory_usage()
    training_time_seconds = end_time - start_time
    training_time_minutes = training_time_seconds / 60
    memory_used = final_memory - initial_memory
    
    # Find the longest token
    longest_token, longest_token_length = find_longest_token(vocab)
    longest_token_display = convert_bytes_to_unicode_for_display(longest_token)
    
    # Save results
    vocab_path = output_dir / "tinystories_vocab.json"
    merges_path = output_dir / "tinystories_merges.txt"
    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)
    
    # Print results
    print(f"\nResults:")
    print(f"- Training completed in {training_time_minutes:.2f} minutes ({training_time_seconds:.2f} seconds)")
    print(f"- Memory used: {memory_used:.2f} GB")
    print(f"- Vocabulary size: {len(vocab)}")
    print(f"- Number of merges: {len(merges)}")
    print(f"- Longest token: '{longest_token_display}' with length {longest_token_length} bytes")
    print(f"- Vocabulary saved to: {vocab_path}")
    print(f"- Merges saved to: {merges_path}")
    
    # Generate the required one-to-two sentence response
    print(f"\nDeliverable response:")
    print(f"The byte-level BPE tokenizer for TinyStories was trained in {training_time_minutes:.2f} minutes using {memory_used:.2f} GB of memory, with the longest token being '{longest_token_display}' ({longest_token_length} bytes), which {' makes sense for natural language text' if longest_token_length < 30 else 'is unusually long and may indicate an issue with the training process'}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the TinyStories dataset")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--output_dir", type=str, default="./tokenizer_output", help="Output directory for vocabulary and merges")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of worker processes for parallel tokenization")
    
    args = parser.parse_args()
    main(args)
