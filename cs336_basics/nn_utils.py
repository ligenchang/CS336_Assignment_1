import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Union, Optional, Tuple
from collections.abc import Iterable as IterableABC
import math
from einops import rearrange, einsum

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
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp

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
    # The function implementation goes here
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

def silu(in_features: Tensor) -> Tensor:
    """
    Applies the Sigmoid Linear Unit (SiLU) function, also known as Swish.
    
    Args:
        in_features (Tensor): Input tensor to apply SiLU on. Shape is arbitrary.
        
    Returns:
        Tensor: Output tensor with the same shape as input.
    """
    # SiLU(x) = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)

def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    use_flash: bool = True
) -> Tensor:
    """
    Computes the scaled dot-product attention as described in the 'Attention is All You Need' paper.
    Uses FlashAttention when available for better memory efficiency and speed.
    
    Args:
        Q (Tensor): Query tensor of shape (..., queries, d_k)
        K (Tensor): Key tensor of shape (..., keys, d_k)
        V (Tensor): Value tensor of shape (..., values, d_v)
        mask (Optional[Tensor]): Optional mask tensor of shape (..., queries, keys)
        use_flash (bool): Whether to use PyTorch's optimized SDPA (FlashAttention when available)
        
    Returns:
        Tensor: Output tensor of shape (..., queries, d_v)
    """
    if use_flash and hasattr(F, 'scaled_dot_product_attention'):
        # Use PyTorch's optimized scaled_dot_product_attention (includes FlashAttention)
        # Convert mask format: PyTorch SDPA expects True for positions to attend to
        attn_mask = None
        if mask is not None:
            attn_mask = mask.bool()
        
        return F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False  # We handle causality with explicit mask
        )
    else:
        # Fallback to manual implementation
        # Get the dimension of the keys
        d_k = K.size(-1)
        
        # Compute the scaled dot-product (Q·K^T / sqrt(d_k))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = softmax(scores, dim=-1)
        
        # Compute the weighted sum (attention weights · V)
        return torch.matmul(attn_weights, V)

def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor
) -> Tensor:
    """
    Applies Rotary Position Embedding (RoPE) to a tensor.
    
    Args:
        d_k (int): Embedding dimension size.
        theta (float): Base value for frequency computation.
        max_seq_len (int): Maximum sequence length.
        in_query_or_key (Tensor): Input tensor of shape (..., seq_len, d_k).
        token_positions (Tensor): Tensor of token positions of shape (..., seq_len).
        
    Returns:
        Tensor: Output tensor with RoPE applied.
    """
    # Create position-dependent rotation angles
    dim_pos = torch.arange(0, d_k, 2, device=in_query_or_key.device).float()
    freqs = 1.0 / (theta ** (dim_pos / d_k))
    
    # Make sure token_positions has the right shape
    if token_positions.dim() < in_query_or_key.dim() - 1:
        # Add batch dimensions if needed
        token_positions = token_positions.view(*([1] * (in_query_or_key.dim() - token_positions.dim() - 1)), *token_positions.shape)
    
    # Compute rotation angles based on token positions
    angles = token_positions.unsqueeze(-1) * freqs
    
    # Compute sine and cosine
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    
    # Prepare sin and cos for rotation
    sin_pos = torch.cat([sin, sin], dim=-1)
    cos_pos = torch.cat([cos, cos], dim=-1)
    
    # Apply rotation:
    # For even indices (0, 2, 4, ...): x_i = x_i * cos - x_(i+1) * sin
    # For odd indices (1, 3, 5, ...): x_i = x_(i-1) * sin + x_i * cos
    # Reshape for easier manipulation
    shape = in_query_or_key.shape
    x = in_query_or_key.view(*shape[:-1], -1, 2)
    
    # Compute rotations
    y1 = x[..., 0] * cos - x[..., 1] * sin
    y2 = x[..., 0] * sin + x[..., 1] * cos
    
    # Reshape back to original shape
    return torch.stack([y1, y2], dim=-1).view(*shape)

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Compute RMS: sqrt(mean(x^2) + eps) over the last dimension (d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Apply RMSNorm: x / RMS(x) * gain
        result = (x / rms) * self.weight
        # Return the result in the original dtype
        return result.to(in_dtype)

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    use_flash: bool = True
) -> Tensor:
    """
    Implements multi-head self-attention as described in the 'Attention is All You Need' paper.
    
    Args:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        q_proj_weight (Tensor): Query projection weights.
        k_proj_weight (Tensor): Key projection weights.
        v_proj_weight (Tensor): Value projection weights.
        o_proj_weight (Tensor): Output projection weights.
        in_features (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        use_flash (bool): Whether to use optimized attention implementation.
        
    Returns:
        Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    batch_size, seq_len, _ = in_features.shape
    head_dim = d_model // num_heads
    
    # Project inputs to queries, keys, and values
    # Using the projection weights as provided by the test
    q = torch.matmul(in_features, q_proj_weight.t())
    k = torch.matmul(in_features, k_proj_weight.t())
    v = torch.matmul(in_features, v_proj_weight.t())
    
    # Reshape for multi-head attention: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    if use_flash and hasattr(F, 'scaled_dot_product_attention'):
        # Use PyTorch's optimized scaled_dot_product_attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # This handles the causal masking efficiently
        )
    else:
        # Fallback to manual implementation
        # Create causal mask to prevent attending to future tokens
        # Shape: [seq_len, seq_len]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=in_features.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(causal_mask, -1e9)  # Apply causal mask
        
        # Apply softmax to get attention weights
        attn_weights = softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
    
    # Reshape back: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # Final linear projection
    output = torch.matmul(attn_output, o_proj_weight.t())
    
    return output

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor = None,
    use_flash: bool = True
) -> Tensor:
    """
    Implements multi-head self-attention with Rotary Position Embedding (RoPE).
    
    Args:
        d_model (int): Dimensionality of the model.
        num_heads (int): Number of attention heads.
        max_seq_len (int): Maximum sequence length.
        theta (float): Base value for RoPE frequency computation.
        q_proj_weight (Tensor): Query projection weights.
        k_proj_weight (Tensor): Key projection weights.
        v_proj_weight (Tensor): Value projection weights.
        o_proj_weight (Tensor): Output projection weights.
        in_features (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        token_positions (Tensor): Tensor of token positions of shape (batch_size, seq_len).
        use_flash (bool): Whether to use optimized attention implementation.
        
    Returns:
        Tensor: Output tensor of shape (batch_size, seq_len, d_model).
    """
    batch_size, seq_len, _ = in_features.shape
    head_dim = d_model // num_heads
    device = in_features.device
    
    # Project inputs to queries, keys, and values
    q = torch.matmul(in_features, q_proj_weight.t())
    k = torch.matmul(in_features, k_proj_weight.t())
    v = torch.matmul(in_features, v_proj_weight.t())
    
    # Reshape for multi-head attention: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Handle token positions for RoPE
    if token_positions is None:
        token_positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Apply RoPE to queries and keys for each head
    for i in range(num_heads):
        q[:, i] = rope(head_dim, theta, max_seq_len, q[:, i], token_positions)
        k[:, i] = rope(head_dim, theta, max_seq_len, k[:, i], token_positions)
    
    if use_flash and hasattr(F, 'scaled_dot_product_attention'):
        # Use PyTorch's optimized scaled_dot_product_attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # This handles the causal masking efficiently
        )
    else:
        # Fallback to manual implementation
        # Create causal mask to prevent attending to future tokens
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(causal_mask, -1e9)  # Apply causal mask
        
        # Apply softmax to get attention weights
        attn_weights = softmax(scores, dim=-1)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
    
    # Reshape back: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    
    # Final linear projection
    output = torch.matmul(attn_output, o_proj_weight.t())
    
    return output

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Initialize with N(0, 2/(din+dout)), truncated to [-3σ, 3σ]
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use einsum for batched linear transformation: (..., in_features), (out_features, in_features) -> (..., out_features)
        return einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Initialize with N(0, 1), truncated to [-3, 3]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Embedding lookup via advanced indexing (einsum doesn't support integer indexing)
        # The weight tensor will automatically be moved to the correct device during forward pass
        return self.weight[token_ids]

def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    in_features: torch.Tensor
) -> torch.Tensor:
    """
    Implements the SwiGLU activation function as used in modern transformer models.
    
    SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
    where SiLU(x) = x * σ(x) = x / (1 + exp(-x))
    
    Args:
        d_model (int): The dimension of the model.
        d_ff (int): The dimension of the feed-forward layer.
        w1_weight (torch.Tensor): The weight matrix for the first projection of shape (d_ff, d_model).
        w2_weight (torch.Tensor): The weight matrix for the second projection of shape (d_model, d_ff).
        w3_weight (torch.Tensor): The weight matrix for the third projection of shape (d_ff, d_model).
        in_features (torch.Tensor): The input tensor of shape (..., d_model).
        
    Returns:
        torch.Tensor: The output tensor of shape (..., d_model).
    """
    # First projections: W1x and W3x
    w1x = einsum(in_features, w1_weight, '... d_model, d_ff d_model -> ... d_ff')
    w3x = einsum(in_features, w3_weight, '... d_model, d_ff d_model -> ... d_ff')
    
    # Apply SiLU activation to W1x and element-wise multiply with W3x
    silu_w1x = silu(w1x)
    gated = silu_w1x * w3x
    
    # Final projection: W2(SiLU(W1x) ⊙ W3x)
    return einsum(gated, w2_weight, '... d_ff, d_model d_ff -> ... d_model')

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict,
    in_features: torch.Tensor
) -> torch.Tensor:
    """
    Implements a single pre-norm transformer block with RoPE.

    Args:
        d_model (int): The dimension of the model.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the feed-forward layer.
        max_seq_len (int): The maximum sequence length.
        theta (float): The RoPE theta parameter.
        weights (dict): A dictionary containing the weights for the transformer block.
        in_features (torch.Tensor): The input tensor of shape (batch, seq_len, d_model).

    Returns:
        torch.Tensor: The output tensor of shape (batch, seq_len, d_model).
    """
    batch_size, seq_len, _ = in_features.shape
    
    # Extract weights
    q_proj_weight = weights["attn.q_proj.weight"]
    k_proj_weight = weights["attn.k_proj.weight"]
    v_proj_weight = weights["attn.v_proj.weight"]
    o_proj_weight = weights["attn.output_proj.weight"]
    ln1_weight = weights["ln1.weight"]
    ffn_w1_weight = weights["ffn.w1.weight"]
    ffn_w2_weight = weights["ffn.w2.weight"]
    ffn_w3_weight = weights["ffn.w3.weight"]
    ln2_weight = weights["ln2.weight"]
    
    # Generate positions for RoPE
    device = in_features.device
    positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # First LayerNorm (pre-norm architecture)
    ln1 = RMSNorm(d_model, 1e-5, device=in_features.device, dtype=in_features.dtype)
    ln1.weight.data.copy_(ln1_weight)
    normed_features = ln1(in_features)
    attn_output = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=normed_features,
        token_positions=positions
    )
    
    # Residual connection
    res1 = in_features + attn_output
    
    # Second LayerNorm (pre-norm for FFN)
    ln2 = RMSNorm(d_model, 1e-5, device=in_features.device, dtype=in_features.dtype)
    ln2.weight.data.copy_(ln2_weight)
    normed_res1 = ln2(res1)
    
    # Feed-forward network with SwiGLU
    ffn_output = swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=ffn_w1_weight,
        w2_weight=ffn_w2_weight,
        w3_weight=ffn_w3_weight,
        in_features=normed_res1
    )
    
    # Final residual connection
    output = res1 + ffn_output
    
    return output

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict,
    in_indices: torch.Tensor
) -> torch.Tensor:
    """
    Implements a transformer language model with RoPE.
    
    Args:
        vocab_size (int): The size of the vocabulary.
        context_length (int): The maximum context length.
        d_model (int): The dimension of the model.
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the feed-forward layer.
        rope_theta (float): The RoPE theta parameter.
        weights (dict): A dictionary containing the weights for the transformer model.
        in_indices (torch.Tensor): The input token indices of shape (batch_size, seq_len).
        
    Returns:
        torch.Tensor: The output logits of shape (batch_size, seq_len, vocab_size).
    """
    batch_size, seq_len = in_indices.shape
    device = in_indices.device
    
    # Embedding using the Embedding class
    embedding_module = Embedding(vocab_size, d_model, device=device)
    embedding_module.weight.data.copy_(weights["token_embeddings.weight"].to(device))
    token_embeddings = embedding_module(in_indices)

    
    # Process through transformer layers
    x = token_embeddings
    for i in range(num_layers):
        # Extract layer weights
        layer_prefix = f"layers.{i}."
        layer_weights = {
            k.replace(layer_prefix, ""): v 
            for k, v in weights.items() 
            if k.startswith(layer_prefix)
        }
        
        # Pass through transformer block
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x
        )
    
    # Final layer norm
    ln_final_weight = weights["ln_final.weight"]
    ln_final = RMSNorm(d_model, 1e-5, device=device, dtype=x.dtype)
    ln_final.weight.data.copy_(ln_final_weight)
    x = ln_final(x)
    
    # Language model head - project to vocabulary
    lm_head_weight = weights["lm_head.weight"]
    logits = torch.matmul(x, lm_head_weight.transpose(0, 1))
    
    return logits
