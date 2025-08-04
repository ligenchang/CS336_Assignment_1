#!/usr/bin/env python3
"""
Test learning rate scheduler continuity after checkpoint resumption.
"""

from cs336_basics.lr_scheduler import get_lr_cosine_schedule

def test_lr_continuity():
    """Test that learning rate continues smoothly when resuming from checkpoint."""
    
    # Original training configuration
    num_steps = 50000
    base_lr = 5e-4
    min_lr = 2e-5
    warmup_iters = int(0.05 * num_steps)  # 2500
    cosine_cycle_iters = num_steps - warmup_iters  # 47500
    
    print(f"Original schedule: num_steps={num_steps}, warmup_iters={warmup_iters}, cosine_cycle_iters={cosine_cycle_iters}")
    
    # Simulate checkpoint at step 38480
    checkpoint_step = 38480
    
    # Learning rate just before and after checkpoint with CORRECT approach
    lr_before = get_lr_cosine_schedule(checkpoint_step - 1, base_lr, min_lr, warmup_iters, cosine_cycle_iters)
    lr_after = get_lr_cosine_schedule(checkpoint_step, base_lr, min_lr, warmup_iters, cosine_cycle_iters)
    lr_next = get_lr_cosine_schedule(checkpoint_step + 1, base_lr, min_lr, warmup_iters, cosine_cycle_iters)
    
    print(f"\nCORRECT approach (using original schedule):")
    print(f"Step {checkpoint_step - 1}: lr={lr_before:.6f}")
    print(f"Step {checkpoint_step}: lr={lr_after:.6f}")
    print(f"Step {checkpoint_step + 1}: lr={lr_next:.6f}")
    print(f"Smooth transition: {abs(lr_after - lr_before) < 0.00001}")
    
    # Show what the INCORRECT approach would do (recalculating based on remaining steps)
    remaining_steps = num_steps - checkpoint_step  # 11520
    wrong_warmup = int(0.05 * remaining_steps)  # 576
    wrong_cosine = remaining_steps - wrong_warmup  # 10944
    
    print(f"\nINCORRECT approach (recalculating for remaining steps):")
    print(f"Remaining steps: {remaining_steps}, wrong_warmup={wrong_warmup}, wrong_cosine={wrong_cosine}")
    
    # This would treat step 0 as the new start
    wrong_lr_0 = get_lr_cosine_schedule(0, base_lr, min_lr, wrong_warmup, wrong_cosine)
    wrong_lr_120 = get_lr_cosine_schedule(120, base_lr, min_lr, wrong_warmup, wrong_cosine)  # ~step 38600
    wrong_lr_320 = get_lr_cosine_schedule(320, base_lr, min_lr, wrong_warmup, wrong_cosine)  # ~step 38800
    
    print(f"Step 0 (relative): lr={wrong_lr_0:.6f}")
    print(f"Step 120 (relative): lr={wrong_lr_120:.6f}")
    print(f"Step 320 (relative): lr={wrong_lr_320:.6f}")
    
    print(f"\nComparison at checkpoint resumption:")
    print(f"Correct continuation: {lr_after:.6f}")
    print(f"Wrong restart approach: {wrong_lr_0:.6f}")
    print(f"Difference: {abs(lr_after - wrong_lr_0):.6f}")

if __name__ == "__main__":
    test_lr_continuity()
