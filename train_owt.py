
import os
import io
import time
import torch
import numpy as np
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import transformer_lm, cross_entropy
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

def validate_model(weights, tokens, batch_size, context_length, device, vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, use_amp=True, num_batches=10):
    """Compute validation loss on a subset of data"""
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
                    loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
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
                loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            
            total_loss += loss.item()
            num_samples += 1
    
    return total_loss / num_samples if num_samples > 0 else float('inf')

def save_checkpoint(weights, optimizer, iteration, best_loss, best_val_loss, out):
    # Only save on rank 0
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    checkpoint = {
        'weights': {k: v.detach().cpu() for k, v in weights.items()},
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'best_loss': best_loss,
        'best_val_loss': best_val_loss
    }
    if out.startswith('s3://'):
        if not s3_available:
            raise RuntimeError('boto3 is required for S3 checkpointing')
        bucket, key = out[5:].split('/', 1)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        boto3.client('s3').upload_fileobj(buffer, bucket, key)
    else:
        torch.save(checkpoint, out)

def main():

    vocab_size = 32000
    context_length = 1024  
    d_model = 768         
    d_ff = 3072           
    num_layers = 16       
    num_heads =  12      
    rope_theta = 10000
    batch_size = 10   
    num_steps = 100000     
    accumulation_steps = 8    

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
    tokens_path = "openwebtext_pretok_tokens.pkl"  # Pre-tokenized OWT data
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "openwebtext_transformer_ckpt.pt")
    curve_path = "openwebtext_learning_curve.npy"
    if rank == 0:
        print(f"Using device: {device}")

    # Optimizer hyperparameters (must be defined before any optimizer usage)
    # base_lr = 1e-4
    # min_lr = 2e-5
    base_lr = 5e-4   # was 1e-4
    min_lr = 2e-5    # was 2e-5

    # Default scheduler parameters
    warmup_iters = int(0.05 * num_steps)
    cosine_cycle_iters = num_steps - warmup_iters

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

    # Try to resume from checkpoint if using CUDA and checkpoint exists
    start_step = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    checkpoint = None
    
    if device.type == "cuda" and checkpoint_path.startswith("s3://") and s3_available:
        bucket, key = checkpoint_path[5:].split('/', 1)
        s3 = boto3.client('s3')
        try:
            buffer = io.BytesIO()
            s3.download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location=device)
            weights = {k: v.to(device).clone().detach().requires_grad_(True) for k, v in checkpoint['weights'].items()}
            optimizer = AdamW(weights.values(), lr=base_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint.get('iteration', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Default to inf if not present
            # Always use original schedule parameters - do not recalculate based on remaining steps
            # The scheduler should continue from where it left off using the original full schedule
            print(f"Resumed from S3 checkpoint at step {start_step} with best_loss {best_loss:.4f}, best_val_loss {best_val_loss:.4f}")
            print(f"Scheduler: using original schedule - warmup_iters={warmup_iters}, cosine_cycle_iters={cosine_cycle_iters}")
            if best_val_loss == float('inf'):
                print("Note: No validation loss in checkpoint - will compute from scratch")
        except Exception as e:
            print(f"Could not load S3 checkpoint: {e}\nStarting from scratch.")
            weights = init_weights()
            optimizer = AdamW(weights.values(), lr=base_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    elif os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            weights = {k: v.to(device).clone().detach().requires_grad_(True) for k, v in checkpoint['weights'].items()}
            optimizer = AdamW(weights.values(), lr=base_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_step = checkpoint.get('iteration', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Default to inf if not present
            # Always use original schedule parameters - do not recalculate based on remaining steps
            # The scheduler should continue from where it left off using the original full schedule
            print(f"Resumed from local checkpoint at step {start_step} with best_loss {best_loss:.4f}, best_val_loss {best_val_loss:.4f}")
            print(f"Scheduler: using original schedule - warmup_iters={warmup_iters}, cosine_cycle_iters={cosine_cycle_iters}")
            if best_val_loss == float('inf'):
                print("Note: No validation loss in checkpoint - will compute from scratch")
        except Exception as e:
            print(f"Could not load local checkpoint: {e}\nStarting from scratch.")
            weights = init_weights()
            optimizer = AdamW(weights.values(), lr=base_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    else:
        weights = init_weights()
        optimizer = AdamW(weights.values(), lr=base_lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)

    # Training loop
    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    log("Starting training on OpenWebText...")
    log(f"Checkpoint will be saved to: {checkpoint_path}")
    log(f"Current best loss to beat: {best_loss:.4f}, best validation loss: {best_val_loss:.4f}")
    losses = []
    # best_loss is already initialized above during checkpoint loading
    use_amp = device.type == "cuda"
    # Use the updated GradScaler API (torch 2.0+)
    try:
        scaler = torch.amp.GradScaler('cuda') if use_amp else None  # Updated API
    except TypeError:
        scaler = torch.amp.GradScaler() if use_amp else None  # Fallback for older PyTorch
    
    # Mixed Precision Strategy:
    # - Forward pass (activations): bfloat16 for memory efficiency and speed
    # - Parameters: float32 for precision in weight updates
    # - Gradients: float32 for stable optimization
    # - Loss computation: float32 for numerical stability
    
    # Wrap weights in DDP if using DDP
    if ddp:
        # Convert weights dict to a torch.nn.Module for DDP
        class WeightsModule(torch.nn.Module):
            def __init__(self, weights):
                super().__init__()
                for k, v in weights.items():
                    self.register_parameter(k.replace('.', '_'), v)
            def forward(self, *args, **kwargs):
                raise NotImplementedError()
        weights_module = WeightsModule(weights)
        weights_module = weights_module.to(device)
        ddp_weights = DDP(weights_module, device_ids=[local_rank])
        # Rebuild weights dict to point to DDP parameters
        weights = {k: getattr(ddp_weights.module, k.replace('.', '_')) for k in weights.keys()}

    optimizer.zero_grad()
    accum_loss = 0.0  # Ensure accum_loss is always initialized
    
    # Initialize data pointer for sequential data access
    data_pointer = 0
    
    # If we resumed from an old checkpoint without validation loss, compute initial validation loss
    if best_val_loss == float('inf') and start_step > 0:
        log("Computing initial validation loss for resumed checkpoint...")
        initial_val_loss = validate_model(
            weights, tokens, batch_size, context_length, device,
            vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, use_amp
        )
        best_val_loss = initial_val_loss
        log(f"Initial validation loss: {initial_val_loss:.4f}")
    
    for step in range(start_step, num_steps):
        for param_group in optimizer.param_groups:
            # Use the global step directly with the original full schedule
            param_group["lr"] = get_lr_cosine_schedule(
                step, base_lr, min_lr, warmup_iters, cosine_cycle_iters
            )
        x, y, data_pointer = get_batch(tokens, batch_size, context_length, device, data_pointer)
        
        # Mixed precision: forward pass in bfloat16, everything else in float32
        if use_amp:
            # Ensure parameters are in float32
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
                # Convert logits back to float32 for loss computation
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
        # Compute main loss
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        # Auxiliary z_loss: encourage log(Z) ~ 0 for stability
        # Z = sum(exp(logits)) over vocab, per token
        # logZ = logsumexp(logits, dim=-1)
        # Take mean over all tokens in batch
        z_loss = 1e-4 * torch.mean(torch.logsumexp(logits, dim=-1))
        loss = loss + z_loss

        # Gradient accumulation - loss and gradients stay in float32
        loss = loss / accumulation_steps
        loss.backward()

        # Accumulate unscaled loss for reporting
        if step % accumulation_steps == 0:
            accum_loss = 0.0
        accum_loss += loss.item()  # This is already divided by accumulation_steps

        if (step + 1) % accumulation_steps == 0:
            # Ensure gradients are in float32 before optimizer step
            for param in weights.values():
                if param.grad is not None:
                    param.grad.data = param.grad.data.float()
            
            optimizer.step()
            optimizer.zero_grad()
            avg_loss = accum_loss  # Since each loss is already divided, sum over accumulation_steps gives average
            losses.append(avg_loss)
            
            # Compute validation loss every 500 steps
            val_loss = None
            if (step + 1) % 500 == 0:
                val_loss = validate_model(
                    weights, tokens, batch_size, context_length, device,
                    vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, use_amp
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            if (step + 1) % 100 == 0:
                val_info = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
                log(f"Step {step + 1}: loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}{val_info}")
            
            # Only check/save checkpoint after optimizer step
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(weights, optimizer, step, best_loss, best_val_loss, checkpoint_path)
                log(f"Best model saved at step {step + 1} with loss {best_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f} to {checkpoint_path}")
            elif val_loss is not None and val_loss < best_val_loss:
                # Save checkpoint if validation loss improved even if training loss didn't
                best_val_loss = val_loss
                val_checkpoint_path = checkpoint_path.replace('.pt', '_best_val.pt') 
                save_checkpoint(weights, optimizer, step, best_loss, best_val_loss, val_checkpoint_path)
                log(f"Best validation model saved at step {step + 1} with val_loss {best_val_loss:.4f} to {val_checkpoint_path}")
    np.save(curve_path, np.array(losses))
    log("Training finished. Learning curve saved.")

if __name__ == "__main__":
    main()

