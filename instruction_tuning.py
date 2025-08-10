"""
Instruction Tuning Module for Supervised Fine-Tuning

This module provides utilities for supervised fine-tuning on instruction-response data,
including data processing, batch creation, and masked loss computation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import pickle
import random
from typing import List, Dict, Tuple, Optional
from cs336_basics.tokenizer import Tokenizer


class InstructionDataProcessor:
    """Process instruction-response data for supervised fine-tuning."""
    
    def __init__(self, tokenizer: Tokenizer, max_length: int = 1024):
        """
        Initialize the instruction data processor.
        
        Args:
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Define special tokens for instruction formatting
        self.instruction_start = "<|instruction|>"
        self.instruction_end = "<|/instruction|>"
        self.response_start = "<|response|>"
        self.response_end = "<|/response|>"
        self.eos_token = "<|endoftext|>"
        
    def load_instruction_dataset(self, filepath: str) -> List[Dict[str, str]]:
        """
        Load instruction dataset from JSON file.
        
        Args:
            filepath: Path to JSON file containing instruction data
            
        Returns:
            List of instruction-response pairs
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is in the expected format
        processed_data = []
        for item in data:
            if isinstance(item, dict) and 'instruction' in item and 'response' in item:
                processed_data.append({
                    'instruction': item['instruction'],
                    'response': item['response']
                })
            elif isinstance(item, dict) and 'input' in item and 'output' in item:
                # Alternative format
                processed_data.append({
                    'instruction': item['input'],
                    'response': item['output']
                })
        
        print(f"Loaded {len(processed_data)} instruction examples from {filepath}")
        return processed_data
    
    def format_instruction_response(self, instruction: str, response: str) -> str:
        """
        Format instruction and response with special tokens.
        
        Args:
            instruction: The instruction text
            response: The response text
            
        Returns:
            Formatted string with special tokens
        """
        return (f"{self.instruction_start}{instruction}{self.instruction_end}"
                f"{self.response_start}{response}{self.response_end}")
    
    def process_single_example(
        self, 
        instruction: str, 
        response: str
    ) -> Tuple[List[int], List[int], List[bool]]:
        """
        Process a single instruction-response pair.
        
        Args:
            instruction: The instruction text
            response: The response text
            
        Returns:
            input_ids: Token IDs for the full sequence
            target_ids: Token IDs for targets (shifted input)
            loss_mask: Boolean mask indicating which tokens to include in loss
        """
        # Format the text
        formatted_text = self.format_instruction_response(instruction, response)
        
        # Tokenize
        full_tokens = self.tokenizer.encode(formatted_text)
        
        # Truncate if too long
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # Create input and target sequences
        input_ids = full_tokens[:-1]  # All tokens except last
        target_ids = full_tokens[1:]  # All tokens except first (shifted)
        
        # Create loss mask - only compute loss on response tokens
        loss_mask = [False] * len(input_ids)
        
        # Find where response starts and ends
        response_start_tokens = self.tokenizer.encode(self.response_start)
        response_end_tokens = self.tokenizer.encode(self.response_end)
        
        # Find response start position
        response_start_pos = None
        for i in range(len(input_ids) - len(response_start_tokens) + 1):
            if input_ids[i:i+len(response_start_tokens)] == response_start_tokens:
                response_start_pos = i + len(response_start_tokens)
                break
        
        # Find response end position
        response_end_pos = None
        if response_start_pos is not None:
            for i in range(response_start_pos, len(input_ids) - len(response_end_tokens) + 1):
                if input_ids[i:i+len(response_end_tokens)] == response_end_tokens:
                    response_end_pos = i
                    break
        
        # Set loss mask for response tokens
        if response_start_pos is not None and response_end_pos is not None:
            for i in range(response_start_pos, response_end_pos):
                if i < len(loss_mask):
                    loss_mask[i] = True
        elif response_start_pos is not None:
            # If no end token found, mask until end of sequence
            for i in range(response_start_pos, len(loss_mask)):
                loss_mask[i] = True
        
        return input_ids, target_ids, loss_mask
    
    def process_dataset(
        self, 
        dataset: List[Dict[str, str]]
    ) -> Tuple[List[List[int]], List[List[int]], List[List[bool]]]:
        """
        Process entire dataset.
        
        Args:
            dataset: List of instruction-response pairs
            
        Returns:
            input_ids_list: List of input token sequences
            target_ids_list: List of target token sequences  
            loss_masks_list: List of loss masks
        """
        input_ids_list = []
        target_ids_list = []
        loss_masks_list = []
        
        for item in dataset:
            try:
                input_ids, target_ids, loss_mask = self.process_single_example(
                    item['instruction'], item['response']
                )
                input_ids_list.append(input_ids)
                target_ids_list.append(target_ids)
                loss_masks_list.append(loss_mask)
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
        
        print(f"Successfully processed {len(input_ids_list)} examples")
        return input_ids_list, target_ids_list, loss_masks_list


def create_instruction_batch(
    input_ids_list: List[List[int]],
    target_ids_list: List[List[int]], 
    loss_masks_list: List[List[bool]],
    batch_size: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a batch from processed instruction data.
    
    Args:
        input_ids_list: List of input token sequences
        target_ids_list: List of target token sequences
        loss_masks_list: List of loss masks
        batch_size: Number of examples per batch
        device: Device to place tensors on
        
    Returns:
        input_batch: [batch_size, seq_len] input token tensor
        target_batch: [batch_size, seq_len] target token tensor
        mask_batch: [batch_size, seq_len] loss mask tensor
    """
    # Sample batch_size examples
    if len(input_ids_list) < batch_size:
        # If we don't have enough examples, sample with replacement
        indices = [random.randint(0, len(input_ids_list) - 1) for _ in range(batch_size)]
    else:
        indices = random.sample(range(len(input_ids_list)), batch_size)
    
    # Get sequences for this batch
    batch_input_ids = [input_ids_list[i] for i in indices]
    batch_target_ids = [target_ids_list[i] for i in indices]
    batch_loss_masks = [loss_masks_list[i] for i in indices]
    
    # Find max length in batch
    max_len = max(len(seq) for seq in batch_input_ids)
    
    # Pad sequences
    input_batch = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    target_batch = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    mask_batch = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        seq_len = len(batch_input_ids[i])
        input_batch[i, :seq_len] = torch.tensor(batch_input_ids[i], dtype=torch.long)
        target_batch[i, :seq_len] = torch.tensor(batch_target_ids[i], dtype=torch.long)
        mask_batch[i, :seq_len] = torch.tensor(batch_loss_masks[i], dtype=torch.bool)
    
    return input_batch, target_batch, mask_batch


def masked_cross_entropy(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute cross-entropy loss with masking.
    
    Args:
        logits: [batch_size, seq_len, vocab_size] model outputs
        targets: [batch_size, seq_len] target token IDs
        mask: [batch_size, seq_len] boolean mask for loss computation
        
    Returns:
        loss: Scalar loss value
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Flatten for loss computation
    flat_logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
    flat_targets = targets.view(-1)  # [batch_size * seq_len]
    flat_mask = mask.view(-1)  # [batch_size * seq_len]
    
    # Compute cross-entropy for all positions
    loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction='none')
    
    # Apply mask
    masked_loss = loss_per_token * flat_mask.float()
    
    # Average over valid tokens
    if flat_mask.sum() > 0:
        return masked_loss.sum() / flat_mask.sum()
    else:
        return masked_loss.sum()  # Return 0 if no valid tokens


def load_instruction_data(filepath: str) -> Tuple[List[List[int]], List[List[int]], List[List[bool]]]:
    """
    Load processed instruction data from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        input_ids_list: List of input token sequences
        target_ids_list: List of target token sequences
        loss_masks_list: List of loss masks
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        return data['input_ids'], data['target_ids'], data['loss_masks']
    elif isinstance(data, (list, tuple)) and len(data) == 3:
        return data[0], data[1], data[2]
    else:
        raise ValueError("Invalid data format in pickle file")


def save_instruction_data(
    input_ids_list: List[List[int]], 
    target_ids_list: List[List[int]], 
    loss_masks_list: List[List[bool]], 
    filepath: str
) -> None:
    """
    Save processed instruction data to pickle file.
    
    Args:
        input_ids_list: List of input token sequences
        target_ids_list: List of target token sequences
        loss_masks_list: List of loss masks
        filepath: Path to save pickle file
    """
    data = {
        'input_ids': input_ids_list,
        'target_ids': target_ids_list,
        'loss_masks': loss_masks_list
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved processed instruction data to {filepath}")


# Helper function for creating sample instruction data
def create_sample_instruction_dataset(num_examples: int = 100) -> List[Dict[str, str]]:
    """
    Create a sample instruction dataset for testing.
    
    Args:
        num_examples: Number of examples to create
        
    Returns:
        List of instruction-response pairs
    """
    sample_instructions = [
        "What is the capital of France?",
        "Explain photosynthesis in simple terms.",
        "How do you make a paper airplane?",
        "What are the benefits of exercise?",
        "Describe the water cycle.",
        "How do computers work?",
        "What is machine learning?",
        "Explain gravity.",
        "How do you cook rice?",
        "What is the meaning of life?"
    ]
    
    sample_responses = [
        "The capital of France is Paris.",
        "Photosynthesis is the process by which plants use sunlight to make food from carbon dioxide and water.",
        "To make a paper airplane, fold a sheet of paper in half lengthwise, then fold the top corners down to form wings.",
        "Exercise helps improve cardiovascular health, strengthens muscles, boosts mood, and maintains healthy weight.",
        "The water cycle involves evaporation from oceans, condensation into clouds, and precipitation as rain or snow.",
        "Computers work by processing binary code through electronic circuits to perform calculations and store data.",
        "Machine learning is a type of AI where computers learn patterns from data to make predictions or decisions.",
        "Gravity is a fundamental force that attracts objects with mass toward each other.",
        "To cook rice, combine 1 cup rice with 2 cups water, bring to boil, then simmer covered for 18-20 minutes.",
        "The meaning of life is a philosophical question that different people answer differently based on their values and beliefs."
    ]
    
    dataset = []
    for i in range(num_examples):
        instruction = sample_instructions[i % len(sample_instructions)]
        response = sample_responses[i % len(sample_responses)]
        dataset.append({
            'instruction': instruction,
            'response': response
        })
    
    return dataset
