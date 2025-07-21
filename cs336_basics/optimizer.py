import math
import torch
from torch.optim import Optimizer
from typing import List, Optional, Callable, Tuple, Dict, Any


class AdamW(Optimizer):
    """
    Implements AdamW algorithm, a variant of Adam with weight decay fix.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2
    ):
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Beta1 must be in [0, 1), got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Beta2 must be in [0, 1), got {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Epsilon must be positive, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
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
                    
                # Get parameters
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                # Get hyperparameters
                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']
                weight_decay = group['weight_decay']
                
                # Update state
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # Decay the first and second moment running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                # Bias correction
                bias_correction1 = 1.0 - beta1 ** state['step']
                bias_correction2 = 1.0 - beta2 ** state['step']
                
                # AdamW: weight decay is applied directly to the parameter 
                # before the optimization step, not to the gradient
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)
                
                # Compute step size
                step_size = lr / bias_correction1
                
                # Apply update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Implements a cosine learning rate schedule with linear warmup.
    
    Following the formula:
    (Warm-up) If t < Tw, then αt = (t/Tw) * αmax.
    (Cosine annealing) If Tw ≤ t ≤ Tc, then αt = αmin + 1/2 * (1 + cos((t-Tw)/(Tc-Tw) * π)) * (αmax-αmin).
    (Post-annealing) If t > Tc, then αt = αmin.
    
    Args:
        it (int): Current iteration number (t).
        max_learning_rate (float): Maximum learning rate (αmax).
        min_learning_rate (float): Minimum learning rate (αmin).
        warmup_iters (int): Number of iterations for linear warmup (Tw).
        cosine_cycle_iters (int): Number of iterations for cosine annealing (Tc).
    
    Returns:
        float: Learning rate for the given iteration.
    """
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # Cosine annealing phase
    if warmup_iters <= it and it <= cosine_cycle_iters:
        cosine_term = math.pi * (it - warmup_iters) / (cosine_cycle_iters -  warmup_iters)
        cosine_factor = 0.5 * (1 + math.cos(cosine_term))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    elif it > cosine_cycle_iters:
        # Post-annealing phase (flat minimum LR)
        return min_learning_rate


    # Post-annealing phase (flat minimum LR)
    return min_learning_rate