#!/usr/bin/env python3
"""
Standalone script to create processed_instructions.pkl

This script processes instruction data and creates the processed pickle file
without running the full training loop.
"""

import os
import argparse
from cs336_basics.tokenizer import Tokenizer
from instruction_tuning import InstructionDataProcessor, save_instruction_data


def create_processed_instructions(
    instruction_data_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    context_length: int = 1024,
    special_tokens: list = None
):
    """
    Create processed instruction data file.
    
    Args:
        instruction_data_path: Path to JSON instruction dataset
        output_path: Path to save processed pickle file
        vocab_path: Path to vocabulary file
        merges_path: Path to merges file
        context_length: Maximum sequence length
        special_tokens: List of special tokens
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<|instruction|>", "<|/instruction|>", "<|response|>", "<|/response|>"]
    
    print(f"Loading tokenizer from {vocab_path} and {merges_path}")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    
    print(f"Creating instruction processor with context_length={context_length}")
    processor = InstructionDataProcessor(tokenizer, context_length)
    
    print(f"Loading instruction dataset from {instruction_data_path}")
    dataset = processor.load_instruction_dataset(instruction_data_path)
    print(f"Loaded {len(dataset)} instruction examples")
    
    print("Processing dataset...")
    input_ids_list, target_ids_list, loss_masks_list = processor.process_dataset(dataset)
    
    print(f"Successfully processed {len(input_ids_list)} examples")
    
    print(f"Saving processed data to {output_path}")
    save_instruction_data(input_ids_list, target_ids_list, loss_masks_list, output_path)
    
    print("✓ Processed instruction data saved successfully!")
    
    # Print some stats
    total_tokens = sum(len(seq) for seq in input_ids_list)
    avg_length = total_tokens / len(input_ids_list) if input_ids_list else 0
    print(f"Statistics:")
    print(f"  - Total examples: {len(input_ids_list)}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average sequence length: {avg_length:.1f} tokens")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Create processed instruction data')
    
    # Required arguments
    parser.add_argument('--instruction_data_path', type=str, required=True,
                       help='Path to instruction dataset JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save processed instruction data (pkl)')
    
    # Tokenizer arguments
    parser.add_argument('--vocab_path', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--merges_path', type=str, required=True,
                       help='Path to merges file')
    
    # Optional arguments
    parser.add_argument('--context_length', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    parser.add_argument('--special_tokens', nargs='+', default=["<|endoftext|>", "<|instruction|>", "<|/instruction|>", "<|response|>", "<|/response|>"],
                       help='Special tokens (default: ["<|endoftext|>", "<|instruction|>", "<|/instruction|>", "<|response|>", "<|/response|>"])')
    
    # Preset configurations
    parser.add_argument('--dataset', type=str, choices=['owt', 'tinystories'],
                       help='Use preset paths for dataset (owt or tinystories)')
    
    args = parser.parse_args()
    
    # Use preset paths if dataset is specified
    if args.dataset == 'owt':
        vocab_path = args.vocab_path if args.vocab_path != 'REQUIRED' else "owt_bpe_vocab.pkl"
        merges_path = args.merges_path if args.merges_path != 'REQUIRED' else "owt_bpe_merges.pkl"
        context_length = 1024
        special_tokens = ["<|endoftext|>", "<|instruction|>", "<|/instruction|>", "<|response|>", "<|/response|>"]
    elif args.dataset == 'tinystories':
        vocab_path = args.vocab_path if args.vocab_path != 'REQUIRED' else "tokenizer_output/tinystories_vocab.json"
        merges_path = args.merges_path if args.merges_path != 'REQUIRED' else "tokenizer_output/tinystories_merges.txt"
        context_length = 256
        special_tokens = ["<|endoftext|>", "<|instruction|>", "<|/instruction|>", "<|response|>", "<|/response|>"]
    else:
        vocab_path = args.vocab_path
        merges_path = args.merges_path
        context_length = args.context_length
        special_tokens = args.special_tokens
    
    # Check if files exist
    if not os.path.exists(args.instruction_data_path):
        print(f"Error: Instruction data file not found: {args.instruction_data_path}")
        return 1
    
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        return 1
    
    if not os.path.exists(merges_path):
        print(f"Error: Merges file not found: {merges_path}")
        return 1
    
    try:
        create_processed_instructions(
            instruction_data_path=args.instruction_data_path,
            output_path=args.output_path,
            vocab_path=vocab_path,
            merges_path=merges_path,
            context_length=context_length,
            special_tokens=special_tokens
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
