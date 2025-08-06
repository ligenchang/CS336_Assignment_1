import torch
import os
import io
import time
from typing import BinaryIO, IO, Union, Optional, Dict, Any, Tuple


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    best_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Enhanced checkpoint saving with support for S3, additional metadata, and error handling.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (Union[str, os.PathLike, BinaryIO, IO[bytes]]): Path or file-like object to serialize the model, optimizer, and iteration to.
        best_loss (Optional[float]): Best training loss achieved so far.
        best_val_loss (Optional[float]): Best validation loss achieved so far.
        additional_metadata (Optional[Dict[str, Any]]): Additional metadata to save in checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'timestamp': time.time(),
    }
    
    # Add optional metadata
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss
    if additional_metadata is not None:
        checkpoint.update(additional_metadata)
    
    # Handle different output types
    if isinstance(out, str) and out.startswith('s3://'):
        _save_checkpoint_to_s3(checkpoint, out)
    else:
        torch.save(checkpoint, out)


def _save_checkpoint_to_s3(checkpoint: Dict[str, Any], s3_path: str) -> None:
    """
    Save checkpoint to S3.
    
    Args:
        checkpoint (Dict[str, Any]): Checkpoint dictionary to save.
        s3_path (str): S3 path in format 's3://bucket/key'.
    """
    try:
        import boto3
        bucket, key = s3_path[5:].split('/', 1)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        boto3.client('s3').upload_fileobj(buffer, bucket, key)
    except ImportError:
        raise ImportError("boto3 is required for S3 checkpoint saving. Install with: pip install boto3")
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint to S3 {s3_path}: {e}")


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> int:
    """
    Original checkpoint loading function for backward compatibility.
    Returns only the iteration number.
    
    Args:
        src (Union[str, os.PathLike, BinaryIO, IO[bytes]]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
        device (Optional[torch.device]): Device to map checkpoint to.
        strict (bool): Whether to strictly enforce that the keys in state_dict match.
    
    Returns:
        int: The iteration number from checkpoint.
    """
    iteration, metadata = load_checkpoint_enhanced(src, model, optimizer, device, strict)
    return iteration


def load_checkpoint_enhanced(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[int, Dict[str, Any]]:
    """
    Enhanced checkpoint loading with support for S3, device mapping, and error handling.
    
    Args:
        src (Union[str, os.PathLike, BinaryIO, IO[bytes]]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
        device (Optional[torch.device]): Device to map checkpoint to.
        strict (bool): Whether to strictly enforce that the keys in state_dict match.
    
    Returns:
        Tuple[int, Dict[str, Any]]: The iteration number and additional metadata from checkpoint.
    """
    # Load checkpoint from different sources
    if isinstance(src, str) and src.startswith('s3://'):
        checkpoint = _load_checkpoint_from_s3(src, device)
    else:
        map_location = device if device is not None else None
        checkpoint = torch.load(src, map_location=map_location)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        raise KeyError("Checkpoint missing 'model_state_dict' key")
    
    # Load optimizer state (check both old and new key names for compatibility)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        raise KeyError("Checkpoint missing optimizer state dict key")
    
    # Extract iteration
    if 'iteration' in checkpoint:
        iteration = checkpoint['iteration']
    else:
        raise KeyError("Checkpoint missing 'iteration' key")
    
    # Extract additional metadata
    metadata = {}
    for key, value in checkpoint.items():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'optimizer', 'iteration']:
            metadata[key] = value
    
    return iteration, metadata


def _load_checkpoint_from_s3(s3_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load checkpoint from S3.
    
    Args:
        s3_path (str): S3 path in format 's3://bucket/key'.
        device (Optional[torch.device]): Device to map checkpoint to.
    
    Returns:
        Dict[str, Any]: Loaded checkpoint dictionary.
    """
    try:
        import boto3
        bucket, key = s3_path[5:].split('/', 1)
        s3 = boto3.client('s3')
        buffer = io.BytesIO()
        s3.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        
        map_location = device if device is not None else None
        return torch.load(buffer, map_location=map_location)
    except ImportError:
        raise ImportError("boto3 is required for S3 checkpoint loading. Install with: pip install boto3")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from S3 {s3_path}: {e}")


def safe_load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Safely load checkpoint with comprehensive error handling.
    
    Args:
        src (Union[str, os.PathLike, BinaryIO, IO[bytes]]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
        device (Optional[torch.device]): Device to map checkpoint to.
        strict (bool): Whether to strictly enforce that the keys in state_dict match.
    
    Returns:
        Optional[Tuple[int, Dict[str, Any]]]: The iteration number and metadata if successful, None if failed.
    """
    try:
        return load_checkpoint_enhanced(src, model, optimizer, device, strict)
    except Exception as e:
        print(f"Failed to load checkpoint from {src}: {e}")
        return None


def checkpoint_exists(path: Union[str, os.PathLike]) -> bool:
    """
    Check if a checkpoint exists at the given path (supports S3).
    
    Args:
        path (Union[str, os.PathLike]): Path to checkpoint.
    
    Returns:
        bool: True if checkpoint exists, False otherwise.
    """
    if isinstance(path, str) and path.startswith('s3://'):
        try:
            import boto3
            bucket, key = path[5:].split('/', 1)
            s3 = boto3.client('s3')
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False
    else:
        return os.path.exists(path)


def create_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_loss: float,
    best_val_loss: float,
    config: Dict[str, Any],
    checkpoint_path: str,
) -> None:
    """
    Create a comprehensive training checkpoint with all relevant information.
    
    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        iteration (int): Current training iteration.
        best_loss (float): Best training loss achieved.
        best_val_loss (float): Best validation loss achieved.
        config (Dict[str, Any]): Training configuration.
        checkpoint_path (str): Path to save checkpoint.
    """
    additional_metadata = {
        'training_config': config,
        'model_config': {
            'vocab_size': getattr(model, 'vocab_size', None),
            'context_length': getattr(model, 'context_length', None),
            'd_model': getattr(model, 'd_model', None),
            'num_layers': getattr(model, 'num_layers', None),
            'num_heads': getattr(model, 'num_heads', None),
            'd_ff': getattr(model, 'd_ff', None),
            'rope_theta': getattr(model, 'rope_theta', None),
        }
    }
    
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=iteration,
        out=checkpoint_path,
        best_loss=best_loss,
        best_val_loss=best_val_loss,
        additional_metadata=additional_metadata
    )


def load_training_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float, float, Dict[str, Any]]:
    """
    Load a comprehensive training checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint.
        model (torch.nn.Module): Model to load state into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        device (torch.device): Device to map checkpoint to.
    
    Returns:
        Tuple[int, float, float, Dict[str, Any]]: iteration, best_loss, best_val_loss, config
    """
    iteration, metadata = load_checkpoint_enhanced(checkpoint_path, model, optimizer, device)
    
    best_loss = metadata.get('best_loss', float('inf'))
    best_val_loss = metadata.get('best_val_loss', float('inf'))
    training_config = metadata.get('training_config', {})
    
    return iteration, best_loss, best_val_loss, training_config


class AdamW(torch.optim.Optimizer):
    """
    Implements AdamW algorithm.
    
    It has been proposed in "Decoupled Weight Decay Regularization" (https://arxiv.org/abs/1711.05101).
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Get grad data
                grad = p.grad.data
                
                # Skip if no gradient
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                # Apply weight decay directly to parameter
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                
                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss
