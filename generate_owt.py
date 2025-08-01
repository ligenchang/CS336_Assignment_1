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
    parser.add_argument('--checkpoint', type=str, default='openwebtext_transformer_ckpt.pt')
    parser.add_argument('--vocab', type=str, default='owt_bpe_vocab.pkl')
    parser.add_argument('--merges', type=str, default='owt_bpe_merges.pkl')
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()

    # Model hyperparameters (must match training)
    vocab_size = 10000
    d_model = 512
    d_ff = 1344
    num_layers = 4
    num_heads = 16
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
        # Get next token (greedy)
        next_token = torch.argmax(logits[0, len(input_ids)-1]).item()
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
