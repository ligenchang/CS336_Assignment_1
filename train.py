"""
Transformer Language Model Training Script

This script provides a robust training pipeline for transformer language models
with support for gradient checkpointing and multiple datasets.
"""

import os
import time
import argparse
import torch
import numpy as np
import pickle

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
            'context_length': 512,    # Shorter context for lower memory
            'd_model': 1792,          # Slightly smaller than previous, between GPT-2 Large and GPT-3 Small
            'd_ff': 7168,             # Slightly smaller FFN
            'num_layers': 36,         # Slightly fewer layers
            'num_heads': 28,          # Fewer heads (must divide d_model)
            'batch_size': 16,         # Keep batch size reasonable for memory
            'num_steps': 160000,      # Keep Chinchilla token budget
            'accumulation_steps': 8,  # Reasonable for this size
            'base_lr': 3e-4,          # Standard LR for GPT-2/3
            'min_lr': 1e-5,           # Proportionally lower min LR
            'max_grad_norm': 1.0,
            'rope_theta': 10000
        }
    
    elif dataset_name == 'wiki':
        return {
            'tokens_path': 'wikipedia_pretok_tokens.pkl',
            'checkpoint_path': 'wikipedia_transformer_ckpt.pt',
            'curve_path': 'wikipedia_learning_curve.npy',
            'vocab_size': 32000,
            'context_length': 1024,
            'd_model': 1024,   # Smaller than GPT-2 Large
            'd_ff': 4096,      # Smaller FFN
            'num_layers': 24,  # Fewer layers
            'num_heads': 16,   # Fewer heads
            'batch_size': 16,   # Smaller batch size
            'num_steps': 160000, # Keep Chinchilla token budget
            'accumulation_steps': 8, # Adjusted for smaller batch
            'base_lr': 3e-4,  # Slightly lower LR
            'min_lr': 1e-5,   # Proportionally lower min LR
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
    parser.add_argument('--dataset', type=str, default='owt', choices=['owt', 'tinystories', "wiki"], 
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
    # MoE options
    parser.add_argument('--use_moe', action='store_true', help='Enable Mixture-of-Experts (MoE) in the transformer FFN')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in MoE layer (if enabled)')
    parser.add_argument('--top_k', type=int, default=2, help='Number of experts to route each token to (if MoE enabled)')
    
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
    
    # Profiling options
    parser.add_argument('--profile', action='store_true', 
                       help='Enable detailed timing profiling')
    parser.add_argument('--torch_profiler', action='store_true', 
                       help='Enable PyTorch profiler (saves to ./profiler_logs)')
    
    # Model optimization options
    parser.add_argument('--compile_model', action='store_true', 
                       help='Enable PyTorch model compilation for better performance')
    parser.add_argument('--auto_batch_size', action='store_true', 
                       help='Automatically find optimal batch size for available GPU memory')
    # Forced checkpoint save
    parser.add_argument('--force_save_once', action='store_true',
                       help='Force a one-time model checkpoint save at the next opportunity')
    # Force data_pointer to 0 (start from beginning of dataset)
    parser.add_argument('--force_data_pointer_zero', action='store_true',
                       help='Force data_pointer to 0 (start from beginning of dataset, ignore checkpoint offset)')
    
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
        'checkpoint_every_n_layers': args.checkpoint_every_n_layers,
        'profile': args.profile,
        'torch_profiler': args.torch_profiler,
        'compile_model': args.compile_model,
        'auto_batch_size': args.auto_batch_size,
        'force_save_once': args.force_save_once,
        'force_data_pointer_zero': args.force_data_pointer_zero,
        # MoE config
        'use_moe': args.use_moe,
        'num_experts': args.num_experts,
        'top_k': args.top_k
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

def init_model(vocab_size, d_model, num_layers, num_heads, d_ff, context_length, rope_theta, device, config):
    """Initialize model using the optimized TransformerLM class.
    MoE support: pass use_moe, num_experts, top_k from config dict."""
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=torch.float32,
        use_moe=config.get('use_moe', False),
        num_experts=config.get('num_experts', 4),
        top_k=config.get('top_k', 2)
    )
    return model



def setup_gpu_optimization():
    """Configure GPU settings for optimal performance."""
    if torch.cuda.is_available():
        # Set memory management environment variables for better memory utilization
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
        
        torch.cuda.empty_cache()
        # Use more aggressive memory fraction to utilize available memory
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
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
        print(f"Memory fraction set to: 95% = {torch.cuda.get_device_properties(0).total_memory * 0.95 / 1024**3:.2f}GB")


def get_gpu_memory_info():
    """Get current GPU memory usage info."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        return f"GPU: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {free:.1f}GB free"
    return "GPU: Not available"




# =============================================================================
# TRAINING LOOP
# =============================================================================

def load_checkpoint_simple(config, device):
    """Simple checkpoint loading using basic serialization."""
    # Always initialize model and optimizer from scratch
    model = init_model(config['vocab_size'], config['d_model'], config['num_layers'], 
                     config['num_heads'], config['d_ff'], config['context_length'], 
                     config['rope_theta'], device, config)
    optimizer = AdamW(model.parameters(), lr=config['base_lr'], betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    
    start_step = 0
    best_loss = float('inf')
    data_pointer = 0  # Default data pointer
    
    checkpoint_path = config['checkpoint_path']
    
    # Try to load checkpoint - handle both local and S3 paths using enhanced functions
    try:
        # Use checkpoint_exists for both S3 and local paths
        print(f"Checking if checkpoint exists at: {checkpoint_path}")
        if checkpoint_exists(checkpoint_path):
            print(f"Checkpoint found! Loading from: {checkpoint_path}")
            try:
                # Try normal load first
                iteration, metadata = load_checkpoint_enhanced(checkpoint_path, model, optimizer, device)
                start_step = iteration + 1
                best_loss = metadata.get('best_loss', float('inf'))
                data_pointer = metadata.get('data_pointer', None)
            except RuntimeError as e:
                if "_orig_mod" in str(e):
                    print("Checkpoint appears to be from a compiled model. Attempting to fix key names...")
                    # Load the raw checkpoint data
                    if checkpoint_path.startswith('s3://'):
                        import boto3
                        import io
                        bucket, key = checkpoint_path.replace('s3://', '').split('/', 1)
                        s3 = boto3.client('s3')
                        buffer = io.BytesIO()
                        s3.download_fileobj(bucket, key, buffer)
                        buffer.seek(0)
                        checkpoint_data = torch.load(buffer, map_location=device)
                    else:
                        checkpoint_data = torch.load(checkpoint_path, map_location=device)

                    # Fix the state dict keys by removing _orig_mod prefix
                    if 'model_state_dict' in checkpoint_data:
                        fixed_state_dict = {}
                        for key, value in checkpoint_data['model_state_dict'].items():
                            if key.startswith('_orig_mod.'):
                                new_key = key[len('_orig_mod.'):]
                                fixed_state_dict[new_key] = value
                            else:
                                fixed_state_dict[key] = value
                        model.load_state_dict(fixed_state_dict)
                        if 'optimizer_state_dict' in checkpoint_data:
                            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                        iteration = checkpoint_data.get('iteration', 0)
                        start_step = iteration + 1
                        # Extract best_loss and data_pointer from top-level keys for compiled model
                        best_loss = checkpoint_data.get('best_loss', float('inf'))
                        data_pointer = checkpoint_data.get('data_pointer', None)
                        print(f"Successfully loaded compiled model checkpoint with fixed keys")
                    else:
                        raise e
                else:
                    raise e

            if data_pointer is None:
                # Calculate data_pointer for legacy checkpoints
                tokens_per_step = config['batch_size'] * config['context_length']
                total_tokens_processed = iteration * tokens_per_step
                with open(config['tokens_path'], "rb") as f:
                    tokens = pickle.load(f)
                total_tokens = len(tokens)
                max_start = total_tokens - config['context_length'] - 1
                data_pointer = total_tokens_processed % total_tokens
                if data_pointer >= max_start:
                    data_pointer = 0
                tokens_per_epoch = total_tokens
                current_epoch = total_tokens_processed // tokens_per_epoch
                tokens_in_current_epoch = total_tokens_processed % tokens_per_epoch
                epoch_progress = (tokens_in_current_epoch / tokens_per_epoch) * 100
                print(f"Legacy checkpoint detected - calculated data_pointer: {data_pointer}")
                print(f"  Total tokens processed: {total_tokens_processed:,}")
                print(f"  Dataset size: {total_tokens:,} tokens")
                print(f"  Tokens per step: {tokens_per_step}")
                print(f"  Training epoch: {current_epoch + 1} (epoch {current_epoch + 1:.1f}, {epoch_progress:.1f}% through current epoch)")
                print(f"  Epochs completed: {current_epoch}, tokens in current epoch: {tokens_in_current_epoch:,}")
            print(f"Resumed from checkpoint at step {start_step} with best_loss {best_loss:.4f}, data_pointer {data_pointer}")
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    except Exception as e:
        print(f"Could not load checkpoint from {checkpoint_path}: {e}\nStarting from scratch.")
    return model, optimizer, start_step, best_loss, data_pointer


def compile_model_if_requested(model, config):
    """Compile model if compilation is requested and supported."""
    if config.get('compile_model', False):
        if hasattr(torch, 'compile'):
            print("Compiling model for optimized performance...")
            try:
                # Use default compilation mode for best balance of compilation time vs performance
                compiled_model = torch.compile(model)
                print("Model compilation successful!")
                return compiled_model
            except Exception as e:
                print(f"Model compilation failed: {e}")
                print("Continuing with uncompiled model...")
                return model
        else:
            print("Model compilation requested but torch.compile not available (requires PyTorch 2.0+)")
            print("Continuing with uncompiled model...")
            return model
    return model


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


    # --- Async streaming token loader ---
    import threading
    import queue

    def token_stream(tokens_path, start_offset=0):
        global_offset = 0
        first_epoch = True
        while True:  # multi-epoch loop
            if first_epoch and start_offset != 0:
                print(f"[DEBUG] token_stream: using start_offset={start_offset}")
            with open(tokens_path, "rb") as f:
                while True:
                    try:
                        doc = pickle.load(f)
                        doc_len = len(doc)
                        if global_offset + doc_len <= start_offset:
                            # Skip this entire chunk
                            global_offset += doc_len
                            continue
                        else:
                            # Yield tokens from start_offset within this chunk
                            start_in_doc = max(0, start_offset - global_offset)
                            for idx in range(start_in_doc, doc_len):
                                yield doc[idx]
                            global_offset += doc_len
                    except EOFError:
                        break
            # After first epoch, reset offsets for subsequent epochs
            global_offset = 0
            start_offset = 0
            first_epoch = False


    def prefetch_tokens(tokens_path, start_offset=0, buffer_size=10_000_000):
        q = queue.Queue(maxsize=2)
        def loader():
            print(f"[DEBUG] prefetch_tokens.loader: starting from start_offset={start_offset}")
            buf = []
            # If resuming from checkpoint, start_offset is the number of tokens already processed
            total_tokens = start_offset
            next_log = ((total_tokens // 100_000_000) + 1) * 100_000_000
            global_offset = start_offset
            for token in token_stream(tokens_path, start_offset=start_offset):
                buf.append(token)
                total_tokens += 1
                if total_tokens >= next_log:
                    print(f"[INFO] Streaming loader: {total_tokens:,} tokens utilized so far.")
                    next_log += 100_000_000
                if len(buf) >= buffer_size:
                    q.put((global_offset, np.array(buf, dtype=np.int32)))
                    global_offset += len(buf)
                    buf = []
            if buf:
                q.put((global_offset, np.array(buf, dtype=np.int32)))
            print("[INFO] Completed one pass through the dataset. Restarting token stream for next epoch.")
        threading.Thread(target=loader, daemon=True).start()
        return q

    print(f"[INFO] Streaming tokens from {config['tokens_path']} with async prefetch...")
    device = setup_device()
    # Load or initialize model and get data_pointer before using it
    model, optimizer, start_step, best_loss, data_pointer = load_checkpoint_simple(config, device)
    # Offload optimizer states to CPU to save GPU memory
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device.type == 'cuda':
                state[k] = v.cpu()
    # Optionally override data_pointer if requested
    if config.get('force_data_pointer_zero', False):
        print("[INFO] --force_data_pointer_zero specified: Forcing data_pointer to 0 (start from beginning of dataset)")
        data_pointer = 0
    start_offset = data_pointer
    tokens_queue = prefetch_tokens(config['tokens_path'], start_offset=start_offset)
    buffer_global_offset, tokens_np = tokens_queue.get()
    print(f"[INFO] Loaded buffer of {len(tokens_np):,} tokens starting at global offset {buffer_global_offset} (streaming mode)")
    buffer_tokens = torch.from_numpy(tokens_np).to(device, dtype=torch.long, non_blocking=True)
    buffer_pointer = start_offset - buffer_global_offset  # position inside buffer
    buffer_len = buffer_tokens.size(0)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    
    # Calculate and log parameter counts for Chinchilla compliance
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = model.token_embeddings.weight.numel() + model.lm_head.weight.numel()
    non_embedding_params = total_params - embedding_params

    # MoE parameter breakdown (if enabled)
    moe_params = 0
    moe_gating_params = 0
    moe_expert_params = 0
    moe_layers = 0
    ffn_params = 0
    if config.get('use_moe', False):
        for layer in model.layers:
            if hasattr(layer, 'use_moe') and layer.use_moe:
                moe_layers += 1
                # Gating: d_model x num_experts + num_experts (bias)
                moe_gating_params += layer.ffn.gate.weight.numel() + layer.ffn.gate.bias.numel()
                # Experts: num_experts * (d_model*2*d_ff + 2*d_ff + d_ff*d_model + d_model)
                # (w1, b1, w2, b2)
                moe_expert_params += (
                    layer.ffn.w1.numel() + layer.ffn.b1.numel() +
                    layer.ffn.w2.numel() + layer.ffn.b2.numel()
                )
            else:
                # Dense FFN (SwiGLU): w1, w2, w3
                if hasattr(layer.ffn, 'w1') and hasattr(layer.ffn, 'w2') and hasattr(layer.ffn, 'w3'):
                    ffn_params += layer.ffn.w1.weight.numel() + layer.ffn.w2.weight.numel() + layer.ffn.w3.weight.numel()
    else:
        # All dense FFN (SwiGLU)
        for layer in model.layers:
            if hasattr(layer.ffn, 'w1') and hasattr(layer.ffn, 'w2') and hasattr(layer.ffn, 'w3'):
                ffn_params += layer.ffn.w1.weight.numel() + layer.ffn.w2.weight.numel() + layer.ffn.w3.weight.numel()

    moe_params = moe_gating_params + moe_expert_params

    print(f"[INFO] Model parameter breakdown:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Embedding parameters (token + lm_head): {embedding_params:,}")
    print(f"  Non-embedding parameters: {non_embedding_params:,}")
    if config.get('use_moe', False):
        print(f"  MoE layers: {moe_layers}")
        print(f"    MoE gating parameters: {moe_gating_params:,}")
        print(f"    MoE expert parameters: {moe_expert_params:,}")
        print(f"    MoE total (gating + experts): {moe_params:,}")
        if ffn_params > 0:
            print(f"    Dense FFN parameters (SwiGLU, non-MoE layers): {ffn_params:,}")
    else:
        print(f"  Dense FFN parameters (SwiGLU): {ffn_params:,}")
    # # Load total number of tokens in the dataset for Chinchilla analysis
    # try:
    #     from tqdm import tqdm
    # except ImportError:
    #     tqdm = None
    # try:
    #     with open(config['tokens_path'], "rb") as f:
    #         doc_count = 0
    #         total_tokens = 0
    #         print("[INFO] Counting total tokens in dataset for Chinchilla analysis (memory efficient)...")
    #         if tqdm is not None:
    #             pbar = tqdm(desc="Counting docs", unit="doc")
    #         else:
    #             pbar = None
    #         while True:
    #             try:
    #                 doc = pickle.load(f)
    #                 total_tokens += len(doc)
    #                 doc_count += 1
    #                 if pbar is not None:
    #                     pbar.update(1)
    #             except EOFError:
    #                 break
    #         if pbar is not None:
    #             pbar.close()
    #         print(f"[INFO] Total docs: {doc_count}, total tokens: {total_tokens:,}")
    # except Exception as e:
    #     print(f"[WARN] Could not load total token count for Chinchilla analysis: {e}")
    #     total_tokens = len(buffer_tokens)
    total_tokens = 8581129216
    print(f"[INFO] Chinchilla scaling analysis:")
    print(f"  Training tokens (total in dataset): {total_tokens:,}")
    print(f"  Chinchilla optimal non-embedding params: ~{total_tokens:,}")
    print(f"  Actual non-embedding params: {non_embedding_params:,}")
    ratio = non_embedding_params / total_tokens if total_tokens > 0 else float('inf')
    print(f"  Parameter/Token ratio: {ratio:.3f} (optimal ≈ 1.0)")
    if 0.8 <= ratio <= 1.2:
        print(f"  ✅ Model size is Chinchilla-optimal!")
    elif ratio < 0.8:
        print(f"  ⚠️  Model is undertrained (too few parameters)")
    else:
        print(f"  ⚠️  Model is overtrained (too many parameters)")
    
    # Compile model if requested
    model = compile_model_if_requested(model, config)

    # Training setup
    optimizer.zero_grad()
    accum_loss = 0.0
    # data_pointer is now loaded from checkpoint or initialized to 0
    use_amp = device.type == "cuda"
    
    # Note: BFloat16 has a wide enough exponent range to avoid gradient underflow,
    # so we don't need GradScaler (which is designed for float16)
    # GradScaler can actually interfere with bfloat16 training
    
    # Scheduler parameters
    warmup_iters = int(0.05 * config['num_steps'])
    cosine_cycle_iters = config['num_steps'] - warmup_iters
    
    def log(msg):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    if config.get('profile', False):
        log("Profiling is ENABLED: timing breakdowns will be printed for every optimizer step.")
    
    # Training info
    log("Starting training...")
    log(f"Training configuration: steps {start_step} to {config['num_steps']} (total: {config['num_steps'] - start_step} remaining)")
    log(f"Gradient checkpointing: {'enabled (every ' + str(config['checkpoint_every_n_layers']) + ' layers)' if config['use_gradient_checkpointing'] else 'disabled'}")
    log(f"Model compilation: {'enabled' if config.get('compile_model', False) else 'disabled'}")
    log(f"Current best loss to beat: {best_loss:.4f}")
    
    losses = []
    
    # Profiling setup
    if config['torch_profiler']:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
        log("PyTorch profiler enabled - traces will be saved to ./profiler_logs")
    else:
        profiler = None
    
    # Main training loop
    # buffer_tokens is already set above
    # buffer_pointer and buffer_len are already set above
    last_best_save_step = -500  # So first possible save is at step 0
    min_best_save_interval = 1000
    force_save_done = False
    progress_log_counter = 0
    for step in range(start_step, config['num_steps']):
        # Regularly clear CUDA cache to help with memory fragmentation
        if torch.cuda.is_available() and step % 100 == 0:
            torch.cuda.empty_cache()
        step_start_time = time.time() if config['profile'] else None

        # Use constant lr for first 90%, then linearly decay to min_lr in last 10%
        progress = (step - start_step) / max(1, config['num_steps'] - start_step)
        if progress < 0.9:
            lr = config['base_lr']
        else:
            # Linear decay from base_lr to min_lr over last 10%
            decay_progress = (progress - 0.9) / 0.1
            lr = config['base_lr'] * (1 - decay_progress) + config['min_lr'] * decay_progress
            lr = max(lr, config['min_lr'])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch, refill buffer if needed
        data_start_time = time.time() if config['profile'] else None
        batch_size = config['batch_size']
        context_length = config['context_length']
        tokens_needed = batch_size * context_length + 1
        # If not enough tokens left in buffer, fetch next buffer
        if buffer_pointer + tokens_needed > buffer_len:
            buffer_global_offset, next_tokens_np = tokens_queue.get()
            buffer_tokens = torch.from_numpy(next_tokens_np).to(device, dtype=torch.long, non_blocking=True)
            buffer_pointer = 0
            buffer_len = buffer_tokens.size(0)
            print(f"[INFO] Switched to new token buffer of {buffer_len:,} tokens.")
        # Slice out the batch
        tokens_slice = buffer_tokens[buffer_pointer:buffer_pointer + tokens_needed]
        x = tokens_slice[:-1].view(batch_size, context_length)
        y = tokens_slice[1:].view(batch_size, context_length)
        buffer_pointer += batch_size * context_length
        global_data_pointer = buffer_global_offset + buffer_pointer
        data_end_time = time.time() if config['profile'] else None
        
        # Forward pass
        forward_start_time = time.time() if config['profile'] else None
        logits = forward_pass(config, model, x, use_amp)
        forward_end_time = time.time() if config['profile'] else None
        
        # Compute loss
        loss_start_time = time.time() if config['profile'] else None
        loss = cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
        z_loss = 1e-4 * torch.mean(torch.logsumexp(logits, dim=-1))
        loss = loss + z_loss
        # Add MoE auxiliary loss if enabled
        if config.get('use_moe', False) and hasattr(model, 'get_moe_loss'):
            moe_loss = model.get_moe_loss()
            # Weight for auxiliary loss can be tuned; 0.01 is a common default
            loss = loss + 0.01 * moe_loss
        loss = loss / config['accumulation_steps']
        loss_end_time = time.time() if config['profile'] else None
        
        # Backward pass
        backward_start_time = time.time() if config['profile'] else None
        loss.backward()
        backward_end_time = time.time() if config['profile'] else None
        
        if step % config['accumulation_steps'] == 0:
            accum_loss = 0.0
        accum_loss += loss.item()
        
        # Gradient update
        if (step + 1) % config['accumulation_steps'] == 0:
            # Gradient clipping and optimization step
            optimizer_start_time = time.time() if config['profile'] else None
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            optimizer.zero_grad()
            optimizer_end_time = time.time() if config['profile'] else None
            
            avg_loss = accum_loss
            losses.append(avg_loss)
            
            # Detailed timing info (always print if profiling is enabled)
            if config['profile']:
                step_total_time = (time.time() - step_start_time) if step_start_time is not None else 0.0
                data_time = (data_end_time - data_start_time) if (data_end_time and data_start_time) else 0.0
                forward_time = (forward_end_time - forward_start_time) if (forward_end_time and forward_start_time) else 0.0
                loss_time = (loss_end_time - loss_start_time) if (loss_end_time and loss_start_time) else 0.0
                backward_time = (backward_end_time - backward_start_time) if (backward_end_time and backward_start_time) else 0.0
                optimizer_time = (optimizer_end_time - optimizer_start_time) if (optimizer_end_time and optimizer_start_time) else 0.0
                log(f"Step {step + 1} timing breakdown:")
                log(f"  Data loading: {data_time:.3f}s ({(data_time/step_total_time*100) if step_total_time else 0:.1f}%)")
                log(f"  Forward pass: {forward_time:.3f}s ({(forward_time/step_total_time*100) if step_total_time else 0:.1f}%)")
                log(f"  Loss compute: {loss_time:.3f}s ({(loss_time/step_total_time*100) if step_total_time else 0:.1f}%)")
                log(f"  Backward pass: {backward_time:.3f}s ({(backward_time/step_total_time*100) if step_total_time else 0:.1f}%)")
                log(f"  Optimizer: {optimizer_time:.3f}s ({(optimizer_time/step_total_time*100) if step_total_time else 0:.1f}%)")
                log(f"  Total step time: {step_total_time:.3f}s")
            
            # Progress logging every 72 optimizer steps since this run started
            progress_log_counter += 1
            if progress_log_counter % 3 == 0 or (step + 1) == config['num_steps']:
                total_steps = config['num_steps']
                finished_steps = step + 1
                percent = 100.0 * finished_steps / max(1, total_steps)
                mem_info = f", {get_gpu_memory_info()}" if torch.cuda.is_available() else ""
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Step {finished_steps}/{total_steps} ({percent:.1f}%): loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, grad_norm={grad_norm:.4f}{mem_info}")
            
       

            # Save best model if a new best is found, but not more than once every 500 steps
            if avg_loss < best_loss and (step + 1 - last_best_save_step >= min_best_save_interval):
                best_loss = avg_loss
                last_best_save_step = step + 1
                save_checkpoint(model, optimizer, step, config['checkpoint_path'], 
                    best_loss=best_loss, additional_metadata={'data_pointer': global_data_pointer})
                log(f"Best model saved at step {step + 1} with loss {best_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f} to {config['checkpoint_path']}")

            # Periodic checkpoint save every 72 optimizer steps since this run started
            if progress_log_counter % 300 == 0:
                periodic_checkpoint_path = config['checkpoint_path'].replace('.pt', f'_step_backup.pt')
                save_checkpoint(model, optimizer, step, periodic_checkpoint_path, 
                    best_loss=avg_loss, additional_metadata={'data_pointer': global_data_pointer})
                log(f"Periodic checkpoint saved at step {step + 1} to {periodic_checkpoint_path}")

            # Force save one-time model checkpoint if requested
            if config.get('force_save_once', False) and not force_save_done:
                best_loss = avg_loss
                force_save_path = config['checkpoint_path']
                save_checkpoint(model, optimizer, step, force_save_path,
                    best_loss=avg_loss, additional_metadata={'data_pointer': global_data_pointer})
                log(f"Force-saved model at step {step + 1} to {force_save_path}")
                log(f"Latest best_loss after force save: {avg_loss:.4f}")
                force_save_done = True
        
        # PyTorch profiler step
        if profiler is not None:
            profiler.step()
            # Stop profiler after a few steps to avoid large files
            if step > start_step + 10:
                profiler.stop()
                profiler = None
                log("PyTorch profiler stopped - check ./profiler_logs for trace files")
    
    # Clean up profiler if still running
    if profiler is not None:
        profiler.stop()
    
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

