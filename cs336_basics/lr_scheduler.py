import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Computes the learning rate for a cosine learning rate schedule with linear warmup.
    
    Implementation of the cosine annealing learning rate schedule used to train LLaMA
    [Touvron et al., 2023], which takes:
    (i) the current iteration t, 
    (ii) the maximum learning rate αmax, 
    (iii) the minimum (final) learning rate αmin, 
    (iv) the number of warm-up iterations Tw, and 
    (v) the number of cosine annealing iterations Tc.
    
    The learning rate at iteration t is defined as:
    
    (Warm-up) If t < Tw, then αt = (t/Tw) * αmax.
    
    (Cosine annealing) If Tw ≤ t ≤ Tc, then αt = αmin + 0.5 * (1 + cos((t-Tw)/(Tc-Tw) * π)) * (αmax-αmin).
    
    (Post-annealing) If t > Tc, then αt = αmin.
    
    Args:
        it (int): Current iteration number (t).
        max_learning_rate (float): Maximum learning rate after warmup (αmax).
        min_learning_rate (float): Minimum learning rate at the end of the schedule (αmin).
        warmup_iters (int): Number of iterations for linear warmup (Tw).
        cosine_cycle_iters (int): Number of iterations for cosine annealing (Tc).
    
    Returns:
        float: Learning rate for the given iteration.
    """
    # (Warm-up) If t < Tw, then αt = (t/Tw) * αmax
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    # (Cosine annealing) If Tw ≤ t ≤ Tc, then αt = αmin + 0.5 * (1 + cos((t-Tw)/(Tc-Tw) * π)) * (αmax-αmin)
    if it < warmup_iters + cosine_cycle_iters:
        cosine_term = ((it - warmup_iters) / cosine_cycle_iters) * math.pi
        cosine_factor = 0.5 * (1 + math.cos(cosine_term))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    
    # (Post-annealing) If t > Tc, then αt = αmin
    return min_learning_rate
