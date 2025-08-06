"""
Transformer Language Model Training Script

This script provides a robust training pipeline for transformer language models
with support for gradient checkpointing and multiple datasets.
"""

import os
import io
import time
import argparse
import torch
import numpy as np
import pickle

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.nn_utils import cross_entropy, TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.lr_scheduler import get_lr_cosine_schedule
from cs336_basics.serialization import save_checkpoint, load_checkpoint_enhanced, checkpoint_exists


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def checkpointed_transformer_lm(model, in_indices, checkpoint_every_n_layers=4):
    """
    Memory-efficient transformer with gradient checkpointing.
    Checkpoints every N layers to trade compute for memory.
    """
    device = in_indices.device
    
    # Embedding (not checkpointed - minimal memory)
    x = model.token_embeddings(in_indices)
    
    # Process layers in checkpointed chunks
    num_layers = len(model.layers)
    for chunk_start in range(0, num_layers, checkpoint_every_n_layers):
        chunk_end = min(chunk_start + checkpoint_every_n_layers, num_layers)
        
        def checkpoint_chunk(x_input, start_idx, end_idx):
            """Process a chunk of transformer layers"""
            x_chunk = x_input
            for i in range(start_idx, end_idx):
                x_chunk = model.layers[i](x_chunk)
            return x_chunk
        
        # Apply gradient checkpointing to this chunk
        x = torch.utils.checkpoint.checkpoint(
            checkpoint_chunk, 
            x, 
            chunk_start, 
            chunk_end,
            use_reentrant=False
        )
    
    # Final layer norm and LM head (not checkpointed - minimal memory)
    x = model.ln_final(x)
    logits = model.lm_head(x)
    
    return logits


# =============================================================================
# DATA UTILITIES
# =============================================================================

def get_batch(tokens, batch_size, context_length, device, data_pointer):
    # ...existing code...
    max_start = len(tokens) - context_length - 1
    batch_x, batch_y = [], []
    for _ in range(batch_size):
        if data_pointer >= max_start:
            data_pointer = 0
        x = tokens[data_pointer:data_pointer + context_length]
        y = tokens[data_pointer + 1:data_pointer + context_length + 1]
        batch_x.append(x)
        batch_y.append(y)
        data_pointer += context_length
    # Use torch.stack for better performance
    x = torch.stack([torch.from_numpy(seq) for seq in batch_x]).to(device, dtype=torch.long, non_blocking=True)
    y = torch.stack([torch.from_numpy(seq) for seq in batch_y]).to(device, dtype=torch.long, non_blocking=True)
    return x, y, data_pointer


# =============================================================================
# CONFIGURATION AND ARGUMENT PARSING
# =============================================================================

def get_dataset_defaults(dataset_name):
    """Get default configuration for specified dataset."""
    if dataset_name == 'owt':
        return {
            'tokens_path': 'openwebtext_pretok_tokens.pkl',
            'checkpoint_path': 'openwebtext_transformer_ckpt.pt',
            'curve_path': 'openwebtext_learning_curve.npy',
            'vocab_size': 32000,
            'context_length': 1024,
            'd_model': 768,
            'd_ff': 2048,
            'num_layers': 24,
            'num_heads': 12,
            'batch_size': 20,
            'num_steps': 60000,
            'accumulation_steps': 16,
            'base_lr': 6e-4,
            'min_lr': 6e-5,
            'max_grad_norm': 1.0,
            'rope_theta': 10000
        }
    elif dataset_name == 'tinystories':
        return {
            'tokens_path': '/Users/michaelli/Downloads/CS336_Assignment_1/tinystories_pretok_tokens.pkl',
            'checkpoint_path': 'tinystories_transformer_ckpt.pt',
            'curve_path': 'tinystories_learning_curve.npy',
            'vocab_size': 10000,
            'context_length': 256,
            'd_model': 512,
            'd_ff': 1344,
            'num_layers': 4,
            'num_heads': 16,
            'batch_size': 32,
            'num_steps': 50000,
            'accumulation_steps': 8,
            'base_lr': 2e-4,
            'min_lr': 2e-5,
            'max_grad_norm': 1.0,
            'rope_theta': 10000
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def parse_args_and_config():
    """Parse command line arguments and build configuration."""
    parser = argparse.ArgumentParser(description='Train Transformer LM on OpenWebText or TinyStories')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='owt', choices=['owt', 'tinystories'], 
                       help='Dataset to use: owt or tinystories')
    
    # File paths
    parser.add_argument('--tokens_path', type=str, default=None, help='Path to pretokenized data (pkl)')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--curve_path', type=str, default=None, help='Learning curve output path')
    
    # Model architecture
    parser.add_argument('--vocab_size', type=int, default=None, help='Override vocab size')
    parser.add_argument('--context_length', type=int, default=None, help='Override context length')
    parser.add_argument('--d_model', type=int, default=None, help='Override d_model')
    parser.add_argument('--d_ff', type=int, default=None, help='Override d_ff')
    parser.add_argument('--num_layers', type=int, default=None, help='Override num_layers')
    parser.add_argument('--num_heads', type=int, default=None, help='Override num_heads')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num_steps', type=int, default=None, help='Override num_steps')
    parser.add_argument('--accumulation_steps', type=int, default=None, help='Override accumulation steps')
    parser.add_argument('--base_lr', type=float, default=None, help='Override base learning rate')
    parser.add_argument('--min_lr', type=float, default=None, help='Override min learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=None, help='Override max grad norm')
    
    # Memory optimization
    parser.add_argument('--use_gradient_checkpointing', action='store_true', 
                       help='Enable gradient checkpointing for memory efficiency')
    parser.add_argument('--checkpoint_every_n_layers', type=int, default=4, 
                       help='Checkpoint every N layers (lower = more memory savings, higher compute cost)')
    
    args = parser.parse_args()
    
    # Get dataset defaults
    default = get_dataset_defaults(args.dataset)
    
    # Build configuration with overrides
    config = {
        'tokens_path': args.tokens_path or os.environ.get('TOKENS_PATH') or default['tokens_path'],
        'checkpoint_path': args.checkpoint_path or os.environ.get('CHECKPOINT_PATH') or default['checkpoint_path'],
        'curve_path': args.curve_path or os.environ.get('CURVE_PATH') or default['curve_path'],
        'vocab_size': args.vocab_size or default['vocab_size'],
        'context_length': args.context_length or default['context_length'],
        'd_model': args.d_model or default['d_model'],
        'd_ff': args.d_ff or default['d_ff'],
        'num_layers': args.num_layers or default['num_layers'],
        'num_heads': args.num_heads or default['num_heads'],
        'batch_size': args.batch_size or default['batch_size'],
        'num_steps': args.num_steps or default['num_steps'],
        'accumulation_steps': args.accumulation_steps or default['accumulation_steps'],
        'base_lr': args.base_lr or default['base_lr'],
        'min_lr': args.min_lr or default['min_lr'],
        'max_grad_norm': args.max_grad_norm or default['max_grad_norm'],
        'rope_theta': default['rope_theta'],
        'use_gradient_checkpointing': args.use_gradient_checkpointing,
        'checkpoint_every_n_layers': args.checkpoint_every_n_layers
    }
    
    return config


# =============================================================================
# DEVICE SETUP
# =============================================================================

def setup_device():
    """Setup device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    return device


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

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

def init_weights_fn(vocab_size, d_model, num_layers, d_ff, device):
    """Initialize model weights with proper scaling (legacy function for compatibility)."""
    weights = {}
    
    # Token embeddings
    emb = torch.empty(vocab_size, d_model, device=device)
    torch.nn.init.trunc_normal_(emb, mean=0.0, std=1.0, a=-3.0, b=3.0)
    weights["token_embeddings.weight"] = torch.nn.Parameter(emb, requires_grad=True)
    
    # Transformer layers
    for i in range(num_layers):
        prefix = f"layers.{i}."
        
        # Attention projections
        for proj in ["attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight", "attn.output_proj.weight"]:
            w = torch.empty(d_model, d_model, device=device)
            std = (2.0 / (d_model + d_model)) ** 0.5
            torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3*std, b=3*std)
            weights[prefix + proj] = torch.nn.Parameter(w, requires_grad=True)
        
        # Layer norms
        weights[prefix + "ln1.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
        weights[prefix + "ln2.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
        
        # Feed-forward weights
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
    
    # Final layer norm
    weights["ln_final.weight"] = torch.nn.Parameter(torch.ones(d_model, device=device), requires_grad=True)
    
    # Language model head
    lm_head = torch.empty(vocab_size, d_model, device=device)
    std_lm = (2.0 / (vocab_size + d_model)) ** 0.5
    torch.nn.init.trunc_normal_(lm_head, mean=0.0, std=std_lm, a=-3*std_lm, b=3*std_lm)
    weights["lm_head.weight"] = torch.nn.Parameter(lm_head, requires_grad=True)
    
    return weights


def setup_gpu_optimization():
    """Configure GPU settings for optimal performance."""
    if torch.cuda.is_available():
        # Set memory management environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        torch.cuda.empty_cache()
        # Enable memory fraction for better allocation
        torch.cuda.set_per_process_memory_fraction(0.9)  # Increased for better utilization
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable optimized attention for better performance
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable tensor caching for better performance
        torch.backends.cudnn.benchmark = True
        
        # Optimize for training
        torch.backends.cudnn.deterministic = False
        
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")


# =============================================================================
# TRAINING LOOP
# =============================================================================

def load_checkpoint_simple(config, device):
    """Simple checkpoint loading using basic serialization."""
    # Always initialize model and optimizer from scratch
    model = init_model(config['vocab_size'], config['d_model'], config['num_layers'], 
                     config['num_heads'], config['d_ff'], config['context_length'], 
                     config['rope_theta'], device)
    optimizer = AdamW(model.parameters(), lr=config['base_lr'], betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    
    start_step = 0
    best_loss = float('inf')
    
    checkpoint_path = config['checkpoint_path']
    
    # Try to load checkpoint - handle both local and S3 paths using enhanced functions
    try:
        # Use checkpoint_exists for both S3 and local paths
        print(f"Checking if checkpoint exists at: {checkpoint_path}")
        if checkpoint_exists(checkpoint_path):
            print(f"Checkpoint found! Loading from: {checkpoint_path}")
            # Use load_checkpoint_enhanced which supports S3
            # Correct parameter order: (src, model, optimizer, device)
            iteration, metadata = load_checkpoint_enhanced(checkpoint_path, model, optimizer, device)
            start_step = iteration + 1
            # Extract best_loss from metadata if available
            best_loss = metadata.get('best_loss', float('inf'))
            print(f"Resumed from checkpoint at step {start_step} with best_loss {best_loss:.4f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    except Exception as e:
        print(f"Could not load checkpoint from {checkpoint_path}: {e}\nStarting from scratch.")
    
    return model, optimizer, start_step, best_loss


def forward_pass(config, model, x, use_amp):
    """Perform forward pass with optional gradient checkpointing."""
    if config['use_gradient_checkpointing']:
        # Use layer-wise gradient checkpointing for memory efficiency
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = checkpointed_transformer_lm(
                    model=model, in_indices=x, 
                    checkpoint_every_n_layers=config['checkpoint_every_n_layers'])
                logits = logits.float()
        else:
            logits = checkpointed_transformer_lm(
                model=model, in_indices=x, 
                checkpoint_every_n_layers=config['checkpoint_every_n_layers'])
    else:
        # Standard forward pass
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                logits = logits.float()
        else:
            logits = model(x)
    
    return logits


def train_loop(config):
    """Main training loop."""
    # Setup environment
    device = setup_device()
    print(f"Using device: {device}")

    setup_gpu_optimization()

    # Load data
    with open(config['tokens_path'], "rb") as f:
        tokens = pickle.load(f)
    tokens = np.array(tokens, dtype=np.int32)

    # Load or initialize model
    model, optimizer, start_step, best_loss = load_checkpoint_simple(config, device)
    model = model.to(device)

    # Training setup
    optimizer.zero_grad()
    accum_loss = 0.0
    data_pointer = 0
    use_amp = device.type == "cuda"
    
    # Note: BFloat16 has a wide enough exponent range to avoid gradient underflow,
    # so we don't need GradScaler (which is designed for float16)
    # GradScaler can actually interfere with bfloat16 training
    
    # Scheduler parameters
    warmup_iters = int(0.05 * config['num_steps'])
    cosine_cycle_iters = config['num_steps'] - warmup_iters
    
    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    
    # Training info
    log("Starting training...")
    log(f"Training configuration: steps {start_step} to {config['num_steps']} (total: {config['num_steps'] - start_step} remaining)")
    log(f"Gradient checkpointing: {'enabled (every ' + str(config['checkpoint_every_n_layers']) + ' layers)' if config['use_gradient_checkpointing'] else 'disabled'}")
    log(f"Current best loss to beat: {best_loss:.4f}")
    
    losses = []
    
    # Main training loop
    for step in range(start_step, config['num_steps']):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = get_lr_cosine_schedule(
                step, config['base_lr'], config['min_lr'], warmup_iters, cosine_cycle_iters)
        
        # Get batch
        x, y, data_pointer = get_batch(tokens, config['batch_size'], config['context_length'], device, data_pointer)
        
        # Forward pass
        logits = forward_pass(config, model, x, use_amp)
        
        # Compute loss
        loss = cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
        z_loss = 1e-4 * torch.mean(torch.logsumexp(logits, dim=-1))
        loss = loss + z_loss
        loss = loss / config['accumulation_steps']
        
        # Backward pass
        loss.backward()
        
        if step % config['accumulation_steps'] == 0:
            accum_loss = 0.0
        accum_loss += loss.item()
        
        # Gradient update
        if (step + 1) % config['accumulation_steps'] == 0:
            # Gradient clipping and optimization step
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()
            
            avg_loss = accum_loss
            losses.append(avg_loss)
            
            # Logging
            if (step + 1) % 100 == 0:
                mem_info = f", GPU_mem={torch.cuda.memory_allocated() / 1024**3:.1f}GB" if torch.cuda.is_available() else ""
                log(f"Step {step + 1}: loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, grad_norm={grad_norm:.4f}{mem_info}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, step, config['checkpoint_path'], best_loss=best_loss)
                log(f"Best model saved at step {step + 1} with loss {best_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f} to {config['checkpoint_path']}")
            
            # Periodic checkpoint save every 2000 steps
            if (step + 1) % 2000 == 0:
                periodic_checkpoint_path = config['checkpoint_path'].replace('.pt', f'_step_{step + 1}.pt')
                save_checkpoint(model, optimizer, step, periodic_checkpoint_path, best_loss=best_loss)
                log(f"Periodic checkpoint saved at step {step + 1} to {periodic_checkpoint_path}")
    
    # Save learning curve
    np.save(config['curve_path'], np.array(losses))
    log("Training finished. Learning curve saved.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    config = parse_args_and_config()
    train_loop(config)


if __name__ == "__main__":
    main()

