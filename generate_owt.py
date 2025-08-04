import argparse
import torch
import numpy as np
import pickle
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import transformer_lm
#python generate_owt.py --prompt "Your prompt here" --length 100

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    weights = {k: v.to(device).clone().detach().requires_grad_(False) for k, v in checkpoint['weights'].items()}
    return weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help='Prompt text to start generation')
    parser.add_argument('--length', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('--checkpoint', type=str, default='/Users/michaelli/Downloads/openwebtext_transformer_ckpt.pt')
    parser.add_argument('--vocab', type=str, default='/Users/michaelli/Downloads/CS336_Assignment_1/owt_bpe_vocab.pkl')
    parser.add_argument('--merges', type=str, default='/Users/michaelli/Downloads/CS336_Assignment_1/owt_bpe_merges.pkl')
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus) sampling probability (default: 1.0 = no filtering)')
    args = parser.parse_args()

    # Model hyperparameters (must match training)
    # vocab_size = 10000
    # d_model = 512
    # d_ff = 1344
    # num_layers = 4
    # num_heads = 16
    # rope_theta = 10000


    # vocab_size = 32000
    # context_length = 512         # You can increase to 768 if memory allows
    # d_model = 512                # Can try up to 768, but 512 is stable
    # d_ff = 2048                  # Typically 4× d_model for better representation
    # num_layers = 10              # 8-12 layers is good balance; 10 here
    # num_heads = 8                # 8 heads fit better with d_model=512 (head_dim=64)
    # rope_theta = 10000

    # vocab_size = 32000
    # context_length = 1024        # Match GPT-2 context length for better fluency
    # d_model = 768                # GPT-2 small hidden size
    # d_ff = 3072                  # 4x d_model, as in GPT-2
    # num_layers = 12              # GPT-2 small depth
    # num_heads = 12               # GPT-2 small heads (head_dim=64)
    # rope_theta = 10000

    vocab_size = 32000
    context_length = 1024        # Increased from 1024 - better for long-form generation
    d_model = 768                # GPT-2 small hidden size
    d_ff = 2048                  # 8/3 x d_model
    num_layers = 8              # GPT-2 small depth
    num_heads =  12              # GPT-2 small heads (head_dim=64)
    rope_theta = 10000
 

    context_length = args.context_length
    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.vocab, args.merges)

    # Load model weights
    weights = load_checkpoint(args.checkpoint, device)

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt)
    if len(prompt_ids) > context_length:
        prompt_ids = prompt_ids[-context_length:]
    generated = list(prompt_ids)

    for _ in range(args.length):
        # Prepare input
        input_ids = generated[-context_length:] if len(generated) > context_length else generated
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        # Forward pass
        logits = transformer_lm(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights=weights,
            in_indices=x
        )
        logits = logits[0, len(input_ids)-1].float()
        # Apply temperature
        if args.temperature != 1.0:
            logits = logits / args.temperature
        probs = torch.softmax(logits, dim=-1)
        # Top-p (nucleus) sampling
        if args.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative_probs > args.top_p
            if torch.any(cutoff):
                last_included = torch.where(cutoff)[0][0].item() + 1
                sorted_probs = sorted_probs[:last_included]
                sorted_indices = sorted_indices[:last_included]
                sorted_probs = sorted_probs / sorted_probs.sum()  # renormalize
            else:
                last_included = len(sorted_probs)
            next_token = sorted_indices[torch.multinomial(sorted_probs, 1).item()].item()
        else:
            # Full softmax sampling
            next_token = torch.multinomial(probs, 1).item()
        # If temperature is very low, fall back to greedy
        if args.temperature == 0:
            next_token = torch.argmax(logits).item()
        generated.append(next_token)
        # Optionally, stop at end-of-text token
        if next_token == tokenizer.byte_to_id.get(b'<|endoftext|>', -1):
            break

    # Decode and print
    out_text = tokenizer.decode(generated)
    print("\n=== Generated Text ===\n")
    print(out_text)

if __name__ == "__main__":
    main()
