
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Iterable, Union, Optional, Tuple
from collections.abc import Iterable as IterableABC
import math
from einops import rearrange, einsum
from torch import nn


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, capacity_factor=1.25, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Gating network
        self.gate = nn.Linear(d_model, num_experts, device=device, dtype=dtype)

        # Expert parameters
        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, 2*d_ff, device=device, dtype=dtype))
        self.b1 = nn.Parameter(torch.zeros(num_experts, 2*d_ff, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_ff, d_model, device=device, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(num_experts, d_model, device=device, dtype=dtype))

        self.reset_parameters()
        self.load_balance_loss = torch.tensor(0.0, device=device, dtype=dtype)

    def reset_parameters(self):
        nn.init.uniform_(self.w1, -1.0 / self.d_model**0.5, 1.0 / self.d_model**0.5)
        nn.init.uniform_(self.w2, -1.0 / self.d_ff**0.5, 1.0 / self.d_ff**0.5)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        B, S, D = x.shape
        N = B * S
        device = x.device
        dtype = x.dtype
        x_flat = x.reshape(N, D)

        # --- Gating ---
        logits = self.gate(x_flat)                   # (N, E)
        probs = F.softmax(logits, dim=-1)           # (N, E)
        topk_scores, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (N, K)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # Efficient top-k >1 support: avoid unnecessary flattening and sorting
        # token_idx_flat: (N, K) -> (N*K,)
        # expert_idx_flat: (N, K) -> (N*K,)
        # gate_flat: (N, K) -> (N*K, 1)
        # x_selected: (N*K, D)
        token_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, self.top_k)
        token_idx_flat = token_idx.reshape(-1)
        expert_idx_flat = topk_idx.reshape(-1)
        gate_flat = topk_scores.reshape(-1, 1)
        x_selected = x_flat[token_idx_flat]

        # --- Capacity mask (vectorized, bincount/cumsum per expert, no sort) ---
        M = expert_idx_flat.size(0)
        # Compute how many tokens are assigned to each expert (for max_capacity)
        expert_counts = torch.bincount(expert_idx_flat, minlength=self.num_experts)
        max_capacity = (self.capacity_factor * expert_counts).ceil().to(torch.long)

        # Compute position in expert for each token (no sort, O(M+E))
        # For each expert, assign a running counter to each token assigned to it
        # Vectorized assignment of position in expert for each token
        # Sort expert_idx_flat to group tokens by expert
        sorted_expert, sort_idx = expert_idx_flat.sort(stable=True)
        # Count tokens per expert
        counts = torch.bincount(sorted_expert, minlength=self.num_experts)
        # Create position indices within each expert
        pos_in_sorted = torch.arange(M, device=device) - torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=counts.dtype), counts[:-1]]), dim=0
        )[sorted_expert]
        # Unsort to original order
        pos_in_expert = torch.empty_like(pos_in_sorted)
        pos_in_expert[sort_idx] = pos_in_sorted
        keep = pos_in_expert < max_capacity[expert_idx_flat]

        token_idx_flat = token_idx_flat[keep]
        expert_idx_flat = expert_idx_flat[keep]
        gate_flat = gate_flat[keep]
        x_selected = x_selected[keep]

        # --- Batched expert MLP ---
        # Gather weights for each token's expert
        w1 = self.w1[expert_idx_flat]  # (M, D, 2*d_ff)
        b1 = self.b1[expert_idx_flat]  # (M, 2*d_ff)
        w2 = self.w2[expert_idx_flat]  # (M, d_ff, D)
        b2 = self.b2[expert_idx_flat]  # (M, D)

        # Preallocate expert output buffer
        y = torch.empty((x_selected.shape[0], D), device=device, dtype=dtype)
        # Use torch.einsum for efficiency
        h = torch.einsum('md,mdh->mh', x_selected.contiguous(), w1.contiguous()) + b1  # (M, 2*d_ff)
        a, b = h.chunk(2, dim=-1)
        h = F.silu(a) * b
        y.copy_(torch.einsum('mf,mfd->md', h.contiguous(), w2.contiguous()) + b2)  # (M, D)
        y.mul_(gate_flat)

        # --- Aggregate outputs (single index_add_) ---
        out = torch.zeros(N, D, device=device, dtype=dtype)
        out.index_add_(0, token_idx_flat, y)

        # --- Load balance loss ---
        expert_usage = probs.mean(dim=0)
        self.load_balance_loss = (expert_usage * (expert_usage.add(1e-9)).log()).sum()

        return out.view(B, S, D).to(dtype)

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
    mask: Optional[Tensor] = None
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

    attn_mask = None
    if mask is not None:
        attn_mask = mask.bool()
    
    return F.scaled_dot_product_attention(
        Q, K, V, 
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False  # We handle causality with explicit mask
    )


class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) as described in Su et al. 2021.
    
    RoPE applies pairwise rotations to query and key vectors based on their positions,
    enabling relative positional encoding without learnable parameters.
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Initialize RoPE module with precomputed frequency values.
        
        Args:
            theta (float): Base value Θ for frequency computation
            d_k (int): Dimension of query and key vectors  
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None): Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Create frequency tensor for positions 0, 2, 4, ..., d_k-2
        dim_pos = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (dim_pos / d_k))
        
        # Register frequencies as a buffer (not saved in state_dict)
        self.register_buffer('freqs', freqs, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., seq_len, d_k)
            token_positions (torch.Tensor): Token positions of shape (..., seq_len)
            
        Returns:
            torch.Tensor: Rotated tensor of same shape as input
        """
        # Make sure token_positions has the right shape
        # if token_positions.dim() < x.dim() - 1:
        #     # Add batch dimensions if needed
        #     print("Alerting!!!!!Reshaping token positions for RoPE")
        #     token_positions = token_positions.view(*([1] * (x.dim() - token_positions.dim() - 1)), *token_positions.shape)
        
        # Compute rotation angles based on token positions
        angles = token_positions.unsqueeze(-1) * self.freqs
        
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
        shape = x.shape
        x_reshaped = x.view(*shape[:-1], -1, 2)
        
        # Compute rotations
        y1 = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
        y2 = x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos
        
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

class MultiHeadSelfAttention(torch.nn.Module):
    """
    Multi-head self-attention module with pre-initialized Linear layers.
    """
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Pre-initialize projection layers
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, in_features: Tensor, use_flash: bool = True) -> Tensor:
        """
        Forward pass for multi-head self-attention.
        
        Args:
            in_features (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            use_flash (bool): Whether to use optimized attention implementation.
            
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = in_features.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)
        
        # Reshape for multi-head attention: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # This handles the causal masking efficiently
        )
        
        # Reshape back: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.output_proj(attn_output)
        
        return output


class MultiHeadSelfAttentionWithRoPE(torch.nn.Module):
    """
    Multi-head self-attention module with RoPE and pre-initialized Linear layers.
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-initialize projection layers
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Pre-initialize RoPE module
        self.rope_module = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
    
    def forward(self, in_features: Tensor, token_positions: Tensor = None, use_flash: bool = True) -> Tensor:
        """
        Forward pass for multi-head self-attention with RoPE.
        
        Args:
            in_features (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            token_positions (Tensor): Tensor of token positions of shape (batch_size, seq_len).
            use_flash (bool): Whether to use optimized attention implementation.
            
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = in_features.shape
        device = in_features.device
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(in_features)
        k = self.k_proj(in_features)
        v = self.v_proj(in_features)
        
        # Reshape for multi-head attention: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
        # Handle token positions for RoPE
        if token_positions is None:
            token_positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Vectorized RoPE: apply to all heads at once for better GPU utilization
        # q, k: [batch_size, num_heads, seq_len, head_dim]
        b, h, s, d = q.shape
        q = q.contiguous().reshape(b * h, s, d)
        k = k.contiguous().reshape(b * h, s, d)
        # Expand token_positions for all heads in the batch
        token_pos_expanded = token_positions.unsqueeze(1).expand(b, h, s).reshape(b * h, s)
        q = self.rope_module(q, token_pos_expanded)
        k = self.rope_module(k, token_pos_expanded)
        q = q.contiguous().reshape(b, h, s, d)
        k = k.contiguous().reshape(b, h, s, d)
        
        # Use PyTorch's optimized scaled_dot_product_attention with causal mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True  # This handles the causal masking efficiently
        )
        
        # Reshape back: [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.output_proj(attn_output)
        
        return output


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Initialize with N(0, 2/(din+dout)), truncated to [-3σ, 3σ]
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use einsum for batched linear transformation: (..., in_features), (out_features, in_features) -> (..., out_features)
        return einsum(x.contiguous(), self.weight.contiguous(), '... d_in, d_out d_in -> ... d_out')


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

class SwiGLU(torch.nn.Module):
    """
    SwiGLU activation function module with pre-initialized Linear layers.
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Pre-initialize Linear layers
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SwiGLU.
        
        Args:
            in_features (torch.Tensor): The input tensor of shape (..., d_model).
            
        Returns:
            torch.Tensor: The output tensor of shape (..., d_model).
        """
        # First projections: W1x and W3x
        w1x = self.w1(in_features)
        w3x = self.w3(in_features)
        
        # Apply SiLU activation to W1x and element-wise multiply with W3x
        silu_w1x = silu(w1x)
        gated = silu_w1x * w3x
        
        # Final projection: W2(SiLU(W1x) ⊙ W3x)
        return self.w2(gated)



class TransformerBlock(torch.nn.Module):
    """
    A single pre-norm transformer block with RoPE and pre-initialized modules.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None, use_moe=False, num_experts=4, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        # Pre-initialize modules
        self.ln1 = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        if use_moe:
            self.ffn = MoELayer(d_model, d_ff, num_experts=num_experts, top_k=top_k, device=device, dtype=dtype)
        else:
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer block.
        
        Args:
            in_features (torch.Tensor): The input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = in_features.shape
        device = in_features.device
        
        # Generate positions for RoPE
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # First LayerNorm (pre-norm architecture)
        normed_features = self.ln1(in_features)
        attn_output = self.attn(normed_features, token_positions=positions)
        
        # Residual connection
        res1 = in_features + attn_output
        
        # Second LayerNorm (pre-norm for FFN)
        normed_res1 = self.ln2(res1)
        
        ffn_output = self.ffn(normed_res1)
        self.moe_loss = 0.0
        # Final residual connection
        output = res1 + ffn_output
        return output


class TransformerLM(torch.nn.Module):
    """
    A complete transformer language model with pre-initialized modules.
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None,
                 use_moe=False, num_experts=4, top_k=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.top_k = top_k
        # Pre-initialize all modules
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype,
                             use_moe=use_moe, num_experts=num_experts, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, 1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer language model with activation checkpointing.
        Returns logits. If MoE is used, you can access the total MoE loss via self.get_moe_loss().
        """
        # Embedding
        x = self.token_embeddings(in_indices)
        # Process through transformer layers with checkpointing
        for layer in self.layers:
            x = layer(x)
        # Final layer norm
        x = self.ln_final(x)
        # Language model head
        logits = self.lm_head(x)
        return logits

    def get_moe_loss(self):
        """Return the sum of MoE auxiliary losses (for load balancing)."""
        if hasattr(self, '_moe_losses') and self._moe_losses:
            return sum(self._moe_losses)
        return 0.0
