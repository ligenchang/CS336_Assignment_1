


# =============================================================================
# Imports and Typing
# =============================================================================
import torch
import os
import io
import time
from typing import BinaryIO, IO, Union, Optional, Dict, Any, Tuple

# =============================================================================
# S3 Utilities
# =============================================================================
def _save_checkpoint_to_s3(checkpoint: Dict[str, Any], s3_path: str) -> None:
    """Save checkpoint to S3."""
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

def _load_checkpoint_from_s3(s3_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load checkpoint from S3."""
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

# =============================================================================
# Checkpoint Existence
# =============================================================================
def checkpoint_exists(path: Union[str, os.PathLike]) -> bool:
    """Check if a checkpoint exists at the given path (supports S3)."""
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

# =============================================================================
# Checkpoint Saving
# =============================================================================
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
    Saves both full and weights-only checkpoints if out is a string path.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'timestamp': time.time(),
    }
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    if best_val_loss is not None:
        checkpoint['best_val_loss'] = best_val_loss
    if additional_metadata is not None:
        checkpoint.update(additional_metadata)
    if isinstance(out, str) and out.startswith('s3://'):
        _save_checkpoint_to_s3(checkpoint, out)
    else:
        torch.save(checkpoint, out)
        if isinstance(out, str):
            weights_path = out.replace(".pt", ".pt.weights")
            torch.save(model.state_dict(), weights_path)

# =============================================================================
# Checkpoint Loading (Full, Enhanced, Any)
# =============================================================================
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
    Returns (iteration, metadata).
    """
    if isinstance(src, str) and src.startswith('s3://'):
        checkpoint = _load_checkpoint_from_s3(src, device)
    else:
        map_location = device if device is not None else None
        checkpoint = torch.load(src, map_location=map_location)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        raise KeyError("Checkpoint missing 'model_state_dict' key")
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        raise KeyError("Checkpoint missing optimizer state dict key")
    if 'iteration' in checkpoint:
        iteration = checkpoint['iteration']
    else:
        raise KeyError("Checkpoint missing 'iteration' key")
    metadata = {}
    for key, value in checkpoint.items():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'optimizer', 'iteration']:
            metadata[key] = value
    return iteration, metadata

def load_any_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[int, float, Optional[int], Dict[str, Any]]:
    """
    Load either a full or weights-only checkpoint.
    Returns (iteration, best_loss, data_pointer, metadata).
    If weights-only, iteration=0, best_loss=inf, data_pointer=None.
    """
    start_step, best_loss, data_pointer = 0, float('inf'), None
    metadata = {}
    if isinstance(src, str) and src.startswith('s3://'):
        checkpoint = _load_checkpoint_from_s3(src, device)
    else:
        map_location = device if device is not None else None
        checkpoint = torch.load(src, map_location=map_location)
    if isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        model.load_state_dict(checkpoint, strict=strict)
        return 0, float('inf'), None, {}
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        raise KeyError("Checkpoint missing 'model_state_dict' key")
    if optimizer is not None:
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    if 'iteration' in checkpoint:
        start_step = checkpoint['iteration'] + 1
    if 'best_loss' in checkpoint:
        best_loss = checkpoint['best_loss']
    if 'data_pointer' in checkpoint:
        data_pointer = checkpoint['data_pointer']
    for key, value in checkpoint.items():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'optimizer', 'iteration', 'best_loss', 'data_pointer']:
            metadata[key] = value
    return start_step, best_loss, data_pointer, metadata


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
        # Only save weights-only file if out is a string path
        if isinstance(out, str):
            weights_path = out.replace(".pt", ".pt.weights")
            torch.save(model.state_dict(), weights_path)


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

