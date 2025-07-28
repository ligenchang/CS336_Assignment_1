import os
import time
import torch
import numpy as np
import pickle
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import transformer_lm, cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.lr_scheduler import get_lr_cosine_schedule

def get_batch(tokens, batch_size, context_length, device):
    idx = np.random.randint(0, len(tokens) - context_length - 1, size=(batch_size,))
    x = np.stack([tokens[i:i+context_length] for i in idx])
    y = np.stack([tokens[i+1:i+context_length+1] for i in idx])
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y

def save_checkpoint(weights, optimizer, iteration, out):
    checkpoint = {
        'weights': {k: v.detach().cpu() for k, v in weights.items()},
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def main():
    # Hyperparameters (same as TinyStories)
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    num_layers = 4
    num_heads = 16
    rope_theta = 10000
    batch_size = 32
    num_steps = 50000
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokens_path = "openwebtext_pretok_tokens.pkl"  # Pre-tokenized OWT data
    checkpoint_path = "openwebtext_transformer_ckpt.pt"
    curve_path = "openwebtext_learning_curve.npy"

    # Load tokenized OWT data
    with open(tokens_path, "rb") as f:
        tokens = pickle.load(f)
    tokens = np.array(tokens, dtype=np.int32)

    # Model weights
    def init_weights():
        weights = {}
        emb = torch.empty(vocab_size, d_model, device=device)
        torch.nn.init.trunc_normal_(emb, mean=0.0, std=0.02, a=-0.04, b=0.04)
        weights["token_embeddings.weight"] = torch.nn.Parameter(emb, requires_grad=True)
        for i in range(num_layers):
            prefix = f"layers.{i}."
            for proj in ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight", "attn.output_proj.weight"]:
                w = torch.empty(d_model, d_model, device=device)
                torch.nn.init.trunc_normal_(w, mean=0.0, std=0.02, a=-0.04, b=0.04)
                weights[prefix + proj] = torch.nn.Parameter(w, requires_grad=True)
            weights[prefix + "ln1.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
            weights[prefix + "ln2.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
            w1 = torch.empty(d_ff, d_model, device=device)
            torch.nn.init.trunc_normal_(w1, mean=0.0, std=0.02, a=-0.04, b=0.04)
            weights[prefix + "ffn.w1.weight"] = torch.nn.Parameter(w1, requires_grad=True)
            w2 = torch.empty(d_model, d_ff, device=device)
            torch.nn.init.trunc_normal_(w2, mean=0.0, std=0.02, a=-0.04, b=0.04)
            weights[prefix + "ffn.w2.weight"] = torch.nn.Parameter(w2, requires_grad=True)
            w3 = torch.empty(d_ff, d_model, device=device)
            torch.nn.init.trunc_normal_(w3, mean=0.0, std=0.02, a=-0.04, b=0.04)
            weights[prefix + "ffn.w3.weight"] = torch.nn.Parameter(w3, requires_grad=True)
        weights["ln_final.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
        lm_head = torch.empty(vocab_size, d_model, device=device)
        torch.nn.init.trunc_normal_(lm_head, mean=0.0, std=0.02, a=-0.04, b=0.04)
        weights["lm_head.weight"] = torch.nn.Parameter(lm_head, requires_grad=True)
        return weights
    weights = init_weights()

    # Optimizer
    base_lr = 9e-4
    optimizer = AdamW(weights.values(), lr=base_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    min_lr = 2e-5
    warmup_iters = int(0.05 * num_steps)
    cosine_cycle_iters = num_steps - warmup_iters

    # Training loop
    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    log("Starting training on OpenWebText...")
    log(f"Checkpoint will be saved to: {checkpoint_path}")
    losses = []
    best_loss = float('inf')
    for step in range(num_steps):
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr_cosine_schedule(
                step, base_lr, min_lr, warmup_iters, cosine_cycle_iters
            )
        x, y = get_batch(tokens, batch_size, context_length, device)
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
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0:
            log(f"Step {step}: loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint(weights, optimizer, step, checkpoint_path)
            log(f"Best model saved at step {step} with loss {best_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f} to {checkpoint_path}")
    np.save(curve_path, np.array(losses))
    log("Training finished. Learning curve saved.")

if __name__ == "__main__":
    main()
