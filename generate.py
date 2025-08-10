import argparse
import torch
import numpy as np
import pickle
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import TransformerLM
#python generate_owt.py --prompt "Your prompt here" --length 100

def softmax_with_temperature(logits, temperature=1.0):
    """Apply temperature scaling and compute softmax."""
    if temperature == 0.0:
        # Greedy decoding - return one-hot at argmax
        argmax_idx = torch.argmax(logits).item()
        probs = torch.zeros_like(logits)
        probs[argmax_idx] = 1.0
        return probs
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Numerically stable softmax
    max_logit = torch.max(scaled_logits)
    exp_logits = torch.exp(scaled_logits - max_logit)
    return exp_logits / torch.sum(exp_logits)

def nucleus_sampling(probs, top_p=1.0):
    """Apply top-p (nucleus) sampling to probability distribution."""
    if top_p >= 1.0:
        # No filtering, return original probabilities
        return probs
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    # Find the cutoff point where cumulative probability exceeds top_p
    # Keep at least the first token (the most likely one)
    cutoff_mask = cumulative_probs > top_p
    if torch.any(cutoff_mask):
        # Find first position where cumsum > top_p
        cutoff_idx = torch.where(cutoff_mask)[0][0].item()
        # Keep tokens up to (but not including) the cutoff
        # But always keep at least the top token
        keep_count = max(1, cutoff_idx)
    else:
        # All tokens have cumulative prob <= top_p, keep all
        keep_count = len(sorted_probs)
    
    # Create filtered probability distribution
    filtered_probs = torch.zeros_like(probs)
    kept_indices = sorted_indices[:keep_count]
    kept_probs = sorted_probs[:keep_count]
    
    # Renormalize the kept probabilities
    kept_probs = kept_probs / torch.sum(kept_probs)
    filtered_probs[kept_indices] = kept_probs
    
    return filtered_probs

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    
    # Handle both old and new checkpoint formats
    if 'weights' in checkpoint:
        # Old format: direct weights dictionary
        weights = {k: v.to(device).clone().detach().requires_grad_(False) for k, v in checkpoint['weights'].items()}
    elif 'model_state_dict' in checkpoint:
        # New format: model state dict from train.py
        weights = {k: v.to(device).clone().detach().requires_grad_(False) for k, v in checkpoint['model_state_dict'].items()}
    else:
        # Try to use the checkpoint directly as weights (legacy format)
        try:
            weights = {k: v.to(device).clone().detach().requires_grad_(False) for k, v in checkpoint.items() 
                      if isinstance(v, torch.Tensor)}
        except Exception as e:
            raise KeyError(f"Could not find model weights in checkpoint. Available keys: {list(checkpoint.keys())}. Error: {e}")
    
    return weights


def main():
    parser = argparse.ArgumentParser(description='Generate text using a trained transformer language model')
    parser.add_argument('--dataset', type=str, default='owt', choices=['owt', 'tinystories'], help='Dataset: owt or tinystories')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text to start generation')
    parser.add_argument('--length', type=int, default=100, help='Maximum number of tokens to generate (default: 100)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (overrides dataset default)')
    parser.add_argument('--vocab', type=str, default=None, help='Path to BPE vocabulary file (overrides dataset default)')
    parser.add_argument('--merges', type=str, default=None, help='Path to BPE merges file (overrides dataset default)')
    parser.add_argument('--context_length', type=int, default=None, help='Context length for generation (overrides dataset default)')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu', help='Device to run on (default: auto-detect)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (0.0=greedy, >1.0=more random, default: 1.0)')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus) sampling threshold (0.0-1.0, default: 1.0=no filtering)')
    args = parser.parse_args()

    # Dataset-specific defaults - match train.py exactly
    if args.dataset == 'owt':
        default_ckpt = '/Users/michaelli/openwebtext_transformer_ckpt_instruction.pt'
        default_vocab = '/Users/michaelli/Downloads/CS336_Assignment_1/owt_bpe_vocab.pkl'
        default_merges = '/Users/michaelli/Downloads/CS336_Assignment_1/owt_bpe_merges.pkl'
        default_num_heads = 12  # From train.py OWT config
        default_context_length = 1024  # From train.py OWT config
    else:
        default_ckpt = 'tinystories_transformer_ckpt.pt'
        default_vocab = '/Users/michaelli/Downloads/CS336_Assignment_1/tinystories_bpe_vocab.pkl'
        default_merges = '/Users/michaelli/Downloads/CS336_Assignment_1/tinystories_bpe_merges.pkl'
        default_num_heads = 16  # From train.py TinyStories config
        default_context_length = 256  # From train.py TinyStories config


    checkpoint = args.checkpoint or default_ckpt
    vocab = args.vocab or default_vocab
    merges = args.merges or default_merges

    # Load model weights early to allow context_length inference
    device = torch.device(args.device)
    weights = load_checkpoint(checkpoint, device)

    def infer_model_params(weights, default_num_heads):
        emb = weights["token_embeddings.weight"]
        vocab_size, d_model = emb.shape
        layer_prefixes = set()
        d_ff = None
        
        # Use dataset-specific default for num_heads instead of trying to infer
        num_heads = default_num_heads
        print(f"[DEBUG] Using dataset-specific num_heads={num_heads} for d_model={d_model}")
        
        for k in weights.keys():
            if k.startswith("layers."):
                parts = k.split(".")
                if len(parts) > 2:
                    layer_prefixes.add(parts[1])
                if d_ff is None and k.endswith("ffn.w1.weight"):
                    d_ff = weights[k].shape[0]
        num_layers = len(layer_prefixes)
        rope_theta = 10000
        context_length = None
        return dict(
            vocab_size=vocab_size,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            rope_theta=rope_theta,
            context_length=context_length
        )

    params = infer_model_params(weights, default_num_heads)
    inferred_context_length = params.get('context_length', None)
    context_length = args.context_length or inferred_context_length or default_context_length


    # Validate arguments
    if args.temperature < 0:
        raise ValueError("Temperature must be non-negative")
    if not (0.0 <= args.top_p <= 1.0):
        raise ValueError("Top-p must be between 0.0 and 1.0")
    if args.length <= 0:
        raise ValueError("Length must be positive")

    # Load tokenizer
    tokenizer = Tokenizer.from_files(vocab, merges)

    # Get end-of-text token ID for stopping
    eos_token_id = tokenizer.byte_to_id.get(b'<|endoftext|>', None)
    if eos_token_id is None:
        print("Warning: <|endoftext|> token not found in tokenizer. Generation may not stop properly.")

    # Load model weights
    weights = load_checkpoint(checkpoint, device)
    # Infer model hyperparameters
    params = infer_model_params(weights, default_num_heads)
    vocab_size = params['vocab_size']
    d_model = params['d_model']
    d_ff = params['d_ff']
    num_layers = params['num_layers']
    num_heads = params['num_heads']
    rope_theta = params['rope_theta']
    print("[INFO] Inferred model parameters from checkpoint:")
    print(f"  vocab_size={vocab_size}, d_model={d_model}, d_ff={d_ff}, num_layers={num_layers}, num_heads={num_heads}, rope_theta={rope_theta}")

    # Create model instance
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=torch.float32
    )
    model.load_state_dict(weights)
    model.eval()  # Set to evaluation mode

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt tokens: {len(prompt_ids)}")
    if len(prompt_ids) > context_length:
        print(f"Warning: Prompt too long ({len(prompt_ids)} tokens), truncating to last {context_length} tokens")
        prompt_ids = prompt_ids[-context_length:]

    generated = list(prompt_ids)
    print(f"Starting generation with prompt: '{args.prompt}'")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print("=" * 50)
    print("\n=== STREAMED OUTPUT ===")
    print(tokenizer.decode(prompt_ids), end="", flush=True)  # Print prompt first

    for step in range(args.length):
        # Prepare input - use only the most recent context_length tokens
        input_ids = generated[-context_length:] if len(generated) > context_length else generated
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Forward pass through transformer
        with torch.no_grad():
            logits = model(x)
        
        # Get logits for the last position (next token prediction)
        next_token_logits = logits[0, len(input_ids)-1].float()
        
        # Apply temperature scaling and compute softmax
        probs = softmax_with_temperature(next_token_logits, args.temperature)
        
        # Apply top-p (nucleus) sampling
        if args.top_p < 1.0:
            probs = nucleus_sampling(probs, args.top_p)
        
        # Sample next token
        if args.temperature == 0.0:
            # Greedy decoding
            next_token = torch.argmax(probs).item()
        else:
            # Stochastic sampling
            next_token = torch.multinomial(probs, 1).item()
        
        generated.append(next_token)
        
        # Stream the new token
        print(tokenizer.decode([next_token]), end="", flush=True)
        
        # Stop if we hit end-of-text token
        if eos_token_id is not None and next_token == eos_token_id:
            print(f"\nGeneration stopped at step {step+1}: <|endoftext|> token generated")
            break

    # Decode and print results
    generated_text = tokenizer.decode(generated)
    prompt_text = tokenizer.decode(prompt_ids)
    
    print("\n" + "=" * 50)
    print("=== GENERATION COMPLETE ===")
    print("=" * 50)
    print(f"Total tokens generated: {len(generated) - len(prompt_ids)}")
    print(f"Total tokens in output: {len(generated)}")
    print("\n=== PROMPT ===")
    print(f'"{prompt_text}"')
    print("\n=== FULL OUTPUT (Prompt + Generated) ===")
    print(f'"{generated_text}"')
    
    # Extract just the generated portion
    if len(generated) > len(prompt_ids):
        generated_only = generated[len(prompt_ids):]
        generated_only_text = tokenizer.decode(generated_only)
        print("\n=== GENERATED TEXT ONLY ===")
        print(f'"{generated_only_text}"')
    else:
        print("\n=== GENERATED TEXT ONLY ===")
        print("(No new tokens generated)")

if __name__ == "__main__":
    main()