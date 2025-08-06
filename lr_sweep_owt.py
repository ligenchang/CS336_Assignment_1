import os
import io
import time
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.lr_scheduler import get_lr_cosine_schedule

# S3 support
try:
    import boto3
    s3_available = True
except ImportError:
    s3_available = False

def get_batch(tokens, batch_size, context_length, device, data_pointer):
    """Get a batch of data sequentially from the dataset"""
    # Ensure we don't go out of bounds
    max_start = len(tokens) - context_length - 1
    
    batch_x = []
    batch_y = []
    
    for _ in range(batch_size):
        if data_pointer >= max_start:
            data_pointer = 0  # Wrap around to start of dataset
        
        x = tokens[data_pointer:data_pointer + context_length]
        y = tokens[data_pointer + 1:data_pointer + context_length + 1]
        
        batch_x.append(x)
        batch_y.append(y)
        
        data_pointer += context_length  # Move forward by context_length tokens
    
    x = torch.tensor(np.stack(batch_x), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(batch_y), dtype=torch.long, device=device)
    
    return x, y, data_pointer

def validate_model(model, tokens, batch_size, context_length, device, vocab_size, use_amp=True, num_batches=10):
    """Validate model using the optimized TransformerLM class."""
    total_loss = 0.0
    num_samples = 0
    
    # Use different range for validation (last 10% of data)
    val_start = int(0.9 * len(tokens))
    val_tokens = tokens[val_start:]
    
    if len(val_tokens) < context_length + 1:
        # Fallback to using a subset of training data
        val_tokens = tokens[-min(len(tokens)//10, 100000):]
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Sample from validation set
            idx = np.random.randint(0, len(val_tokens) - context_length - 1, size=(batch_size,))
            x = np.stack([val_tokens[i:i+context_length] for i in idx])
            y = np.stack([val_tokens[i+1:i+context_length+1] for i in idx])
            x = torch.tensor(x, dtype=torch.long, device=device)
            y = torch.tensor(y, dtype=torch.long, device=device)
            
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    logits = logits.float()
                    loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            else:
                logits = model(x)
                loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            total_loss += loss.item()
            num_samples += 1
    
    return total_loss / num_samples if num_samples > 0 else float('inf')

def init_model(vocab_size, d_model, num_layers, num_heads, d_ff, context_length, rope_theta, device):
    """Initialize model using the optimized TransformerLM class."""
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
    return model

def init_weights(vocab_size, d_model, num_layers, d_ff, device):
    """Initialize model weights"""
    weights = {}
    # Embedding: N(0, 1), truncated to [-3, 3]
    emb = torch.empty(vocab_size, d_model, device=device)
    torch.nn.init.trunc_normal_(emb, mean=0.0, std=1.0, a=-3.0, b=3.0)
    weights["token_embeddings.weight"] = torch.nn.Parameter(emb, requires_grad=True)
    
    for i in range(num_layers):
        prefix = f"layers.{i}."
        # Linear weights: N(0, 2/(din+dout)), truncated to [-3σ, 3σ]
        for proj in ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight", "attn.output_proj.weight"]:
            w = torch.empty(d_model, d_model, device=device)
            std = (2.0 / (d_model + d_model)) ** 0.5
            torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)
            weights[prefix + proj] = torch.nn.Parameter(w, requires_grad=True)
        weights[prefix + "ln1.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
        weights[prefix + "ln2.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
        # FFN weights
        w1 = torch.empty(d_ff, d_model, device=device)
        std1 = (2.0 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(w1, mean=0.0, std=std1, a=-3*std1, b=3*std1)
        weights[prefix + "ffn.w1.weight"] = torch.nn.Parameter(w1, requires_grad=True)
        w2 = torch.empty(d_model, d_ff, device=device)
        std2 = (2.0 / (d_model + d_ff)) ** 0.5
        torch.nn.init.trunc_normal_(w2, mean=0.0, std=std2, a=-3*std2, b=3*std2)
        weights[prefix + "ffn.w2.weight"] = torch.nn.Parameter(w2, requires_grad=True)
        w3 = torch.empty(d_ff, d_model, device=device)
        std3 = (2.0 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(w3, mean=0.0, std=std3, a=-3*std3, b=3*std3)
        weights[prefix + "ffn.w3.weight"] = torch.nn.Parameter(w3, requires_grad=True)
    
    weights["ln_final.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
    # LM head: N(0, 2/(din+dout)), din=d_model, dout=vocab_size
    lm_head = torch.empty(vocab_size, d_model, device=device)
    std_lm = (2.0 / (vocab_size + d_model)) ** 0.5
    torch.nn.init.trunc_normal_(lm_head, mean=0.0, std=std_lm, a=-3*std_lm, b=3*std_lm)
    weights["lm_head.weight"] = torch.nn.Parameter(lm_head, requires_grad=True)
    return weights

def train_with_lr(base_lr, min_lr, tokens, device, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, 
                  batch_size, num_steps, accumulation_steps, max_grad_norm, warmup_iters, cosine_cycle_iters, 
                  rank=0, log_interval=100, val_interval=500):
    """Train model with specific learning rate and return training/validation curves"""
    
    def log(msg):
        if rank == 0:
            print(f"[LR={base_lr:.2e}] {msg}")
    
    log(f"Starting training with base_lr={base_lr:.2e}, min_lr={min_lr:.2e}")
    
    # Initialize fresh weights for this LR
    weights = init_weights(vocab_size, d_model, num_layers, d_ff, device)
    optimizer = AdamW(weights.values(), lr=base_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    
    losses = []
    val_losses = []
    val_steps = []
    use_amp = device.type == "cuda"
    
    # Use the updated GradScaler API (torch 2.0+)
    try:
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
    except TypeError:
        scaler = torch.amp.GradScaler() if use_amp else None
    
    optimizer.zero_grad()
    accum_loss = 0.0
    data_pointer = 0
    
    # Track if training diverged
    diverged = False
    last_loss = float('inf')
    divergence_threshold = 10.0  # If loss > 10, consider diverged
    
    for step in range(num_steps):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr_cosine_schedule(
                step, base_lr, min_lr, warmup_iters, cosine_cycle_iters
            )

        x, y, data_pointer = get_batch(tokens, batch_size, context_length, device, data_pointer)

        # Debug: print batch stats for first step
        if step == 0 and rank == 0:
            print("[DEBUG] Batch x stats:", x.shape, x.dtype, "min:", x.min().item(), "max:", x.max().item())
            print("[DEBUG] Batch y stats:", y.shape, y.dtype, "min:", y.min().item(), "max:", y.max().item())
            for k, v in weights.items():
                print(f"[DEBUG] Weight {k}: mean={v.data.mean().item():.4f}, std={v.data.std().item():.4f}, min={v.data.min().item():.4f}, max={v.data.max().item():.4f}")

        # Forward pass
        if use_amp:
            for param in weights.values():
                param.data = param.data.float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
                logits = logits.float()
        else:
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

        # Compute loss
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        z_loss = 1e-4 * torch.mean(torch.logsumexp(logits, dim=-1))
        loss = loss + z_loss

        # Debug: print initial loss before divergence check
        if step == 0 and rank == 0:
            print(f"[DEBUG] Initial loss (step 0): {loss.item():.4f}")

        # Check for divergence
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > divergence_threshold:
            log(f"Training diverged at step {step + 1} with loss {loss.item():.4f}")
            diverged = True
            break
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if step % accumulation_steps == 0:
            accum_loss = 0.0
        accum_loss += loss.item()
        
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            for param in weights.values():
                if param.grad is not None:
                    param.grad.data = param.grad.data.float()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(weights.values(), max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            
            avg_loss = accum_loss
            losses.append(avg_loss)
            last_loss = avg_loss
            
            # Validation every val_interval steps
            if (step + 1) % val_interval == 0:
                val_loss = validate_model(
                    weights, tokens, batch_size, context_length, device,
                    vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, use_amp, num_batches=5
                )
                val_losses.append(val_loss)
                val_steps.append(step + 1)
                
                if (step + 1) % log_interval == 0:
                    log(f"Step {step + 1}: loss={avg_loss:.4f}, val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, grad_norm={grad_norm:.4f}")
            elif (step + 1) % log_interval == 0:
                log(f"Step {step + 1}: loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, grad_norm={grad_norm:.4f}")
    
    final_loss = last_loss if not diverged else float('inf')
    final_val_loss = val_losses[-1] if val_losses and not diverged else float('inf')
    
    log(f"Finished: final_loss={final_loss:.4f}, final_val_loss={final_val_loss:.4f}, diverged={diverged}")
    
    return {
        'base_lr': base_lr,
        'min_lr': min_lr,
        'losses': losses,
        'val_losses': val_losses,
        'val_steps': val_steps,
        'final_loss': final_loss,
        'final_val_loss': final_val_loss,
        'diverged': diverged,
        'total_steps': len(losses) * accumulation_steps
    }

def main():
    # Model configuration (same as train_owt.py)
    # vocab_size = 32000
    # context_length = 1024
    # d_model = 768
    # d_ff = 3072
    # num_layers = 16
    # num_heads = 12
    # rope_theta = 10000
    # batch_size = 10
    # accumulation_steps = 8
    # max_grad_norm = 1.0

    # Use a smaller model and batch for debugging
    vocab_size = 32000
    context_length = 32
    d_model = 128
    d_ff = 512
    num_layers = 2
    num_heads = 4
    rope_theta = 10000
    batch_size = 2
    accumulation_steps = 1
    max_grad_norm = 1.0
    
    # Shorter training for sweep (adjust based on compute budget)
    num_steps = 100  # Reduced from 100k for faster sweep
    
    # Learning rate sweep configuration
    # Strategy: Log-spaced search around the baseline, then zoom in on promising region
    learning_rates = [
        5e-5,   # Very conservative
        1e-4,   # Conservative baseline
        2e-4,   # Original baseline from train_owt.py
        3e-4,   # Slightly aggressive
        5e-4,   # More aggressive
        7e-4,   # Getting risky
        1e-3,   # Very aggressive
        2e-3,   # Likely to diverge
    ]
    
    # Minimum LR is typically 10-20% of base LR
    min_lr_ratio = 0.1  # min_lr = base_lr * min_lr_ratio

    # DDP setup
    ddp = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        ddp = True
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        rank = 0

    # Load tokenized data
    tokens_path = "/Users/michaelli/Downloads/CS336_Assignment_1/openwebtext_pretok_tokens.pkl"
    with open(tokens_path, "rb") as f:
        tokens = pickle.load(f)
    tokens = np.array(tokens, dtype=np.int32)

    # Scheduler parameters
    warmup_iters = int(0.05 * num_steps)
    cosine_cycle_iters = num_steps - warmup_iters

    # Run sweep
    results = []
    for i, base_lr in enumerate(learning_rates):
        min_lr = base_lr * min_lr_ratio
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Experiment {i+1}/{len(learning_rates)}: LR={base_lr:.2e}")
            print(f"{'='*60}")
        result = train_with_lr(
            base_lr=base_lr,
            min_lr=min_lr,
            tokens=tokens,
            device=device,
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            batch_size=batch_size,
            num_steps=num_steps,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
            rank=rank
        )
        results.append(result)
    if rank == 0:
        # Generate summary and plots
        print(f"\n{'='*80}")
        print("LEARNING RATE SWEEP SUMMARY")
        print(f"{'='*80}")
        print(f"{'LR':<10} {'Min LR':<10} {'Final Loss':<12} {'Final Val':<12} {'Diverged':<10} {'Steps':<8}")
        print(f"{'-'*70}")
        best_lr = None
        best_val_loss = float('inf')
        for result in results:
            diverged_str = "YES" if result['diverged'] else "NO"
            final_loss_str = f"{result['final_loss']:.4f}" if not result['diverged'] else "INF"
            final_val_str = f"{result['final_val_loss']:.4f}" if not result['diverged'] else "INF"
            print(f"{result['base_lr']:<10.2e} {result['min_lr']:<10.2e} {final_loss_str:<12} {final_val_str:<12} {diverged_str:<10} {result['total_steps']:<8}")
            if not result['diverged'] and result['final_val_loss'] < best_val_loss:
                best_val_loss = result['final_val_loss']
                best_lr = result['base_lr']
        print(f"{'-'*70}")
        if best_lr is not None:
            print(f"Best LR: {best_lr:.2e} (val_loss: {best_val_loss:.4f})")
        else:
            print("All learning rates diverged!")
        # Plot learning curves
        plt.figure(figsize=(15, 10))
        # Training loss curves
        plt.subplot(2, 2, 1)
        for result in results:
            if not result['diverged'] and result['losses']:
                steps = np.arange(1, len(result['losses']) + 1) * accumulation_steps
                plt.plot(steps, result['losses'], label=f"LR={result['base_lr']:.2e}")
        plt.xlabel('Training Steps')
        plt.ylabel('Training Loss')
        plt.title('Training Loss vs Steps')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        # Validation loss curves
        plt.subplot(2, 2, 2)
        for result in results:
            if not result['diverged'] and result['val_losses']:
                plt.plot(result['val_steps'], result['val_losses'], label=f"LR={result['base_lr']:.2e}", marker='o')
        plt.xlabel('Training Steps')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss vs Steps')
        plt.legend()
        plt.grid(True)
        # Final losses vs LR
        plt.subplot(2, 2, 3)
        lrs = []
        final_losses = []
        final_val_losses = []
        for result in results:
            if not result['diverged']:
                lrs.append(result['base_lr'])
                final_losses.append(result['final_loss'])
                final_val_losses.append(result['final_val_loss'])
        if lrs:
            plt.semilogx(lrs, final_losses, 'bo-', label='Training Loss')
            plt.semilogx(lrs, final_val_losses, 'ro-', label='Validation Loss')
            plt.xlabel('Learning Rate')
            plt.ylabel('Final Loss')
            plt.title('Final Loss vs Learning Rate')
            plt.legend()
            plt.grid(True)
        # Learning rate schedule for best LR
        plt.subplot(2, 2, 4)
        if best_lr is not None:
            steps = np.arange(num_steps)
            lrs_schedule = [get_lr_cosine_schedule(step, best_lr, best_lr * min_lr_ratio, warmup_iters, cosine_cycle_iters) for step in steps]
            plt.plot(steps, lrs_schedule)
            plt.xlabel('Training Steps')
            plt.ylabel('Learning Rate')
            plt.title(f'LR Schedule for Best LR ({best_lr:.2e})')
            plt.grid(True)
        plt.tight_layout()
        plt.show()
        print(f"\nResults visualized.")
        # Hyperparameter search strategy explanation
        print(f"\n{'='*80}")
        print("HYPERPARAMETER SEARCH STRATEGY")
        print(f"{'='*80}")
        print("1. Log-spaced search: Tested LRs spanning 2 orders of magnitude")
        print("2. Conservative to aggressive: Started with 5e-5, ended with 2e-3")
        print("3. Baseline anchor: Included 2e-4 (current working LR)")
        print("4. Divergence detection: Stop training if loss > 10 or NaN/Inf")
        print("5. Validation tracking: Measure generalization, not just training loss")
        print("6. Reduced steps: 10k steps per LR for faster iteration")
        print("7. Next steps: Zoom in around best LR with finer grid")

if __name__ == "__main__":
    main()
