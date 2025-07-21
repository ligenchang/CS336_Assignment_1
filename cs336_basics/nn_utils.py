import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Union, Optional
from collections.abc import Iterable as IterableABC

def softmax(in_features: Tensor, dim: int) -> Tensor:
    """
    Implementation of the softmax function along a specified dimension.
    
    Args:
        in_features (Tensor): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.
        
    Returns:
        Tensor: Tensor with the same shape as `in_features` with the output of
               softmax normalizing the specified `dim`.
    """
    # Shift for numerical stability (helps prevent overflow)
    shifted_input = in_features - in_features.max(dim=dim, keepdim=True)[0]
    
    # Compute exponentials
    exp_x = torch.exp(shifted_input)
    
    # Normalize
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """
    Compute the cross-entropy loss between inputs and targets.
    
    Args:
        inputs (Tensor): Unnormalized logits of shape (batch_size, vocab_size)
        targets (Tensor): Class indices of shape (batch_size,)
        
    Returns:
        Tensor: Average cross-entropy loss across examples.
    """
    # Get the batch size and vocabulary size
    batch_size, vocab_size = inputs.shape
    
    # Apply softmax to get probabilities
    log_probs = F.log_softmax(inputs, dim=-1)
    
    # Gather the log probabilities corresponding to the target classes
    nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Return the average loss
    return nll_loss.mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Clip gradients of parameters to have a maximum L2 norm.
    
    Args:
        parameters (Iterable[torch.nn.Parameter]): Collection of trainable parameters.
        max_l2_norm (float): Maximum L2 norm value.
    """
    # Filter parameters with gradients
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    
    if not parameters_with_grad:
        return
    
    # Calculate the total norm of all gradients
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters_with_grad]), 2
    )
    
    # Calculate the scaling factor
    clip_coef = max_l2_norm / (total_norm + 1e-8)
    
    # If the total norm is larger than the maximum allowed norm, scale all gradients
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.detach().mul_(clip_coef)
