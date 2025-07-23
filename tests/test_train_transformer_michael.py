import torch
import pytest
from cs336_basics.nn_utils import (
    transformer_lm,
    embedding,
    linear,
    swiglu,
    rmsnorm,
    multihead_self_attention,
    multihead_self_attention_with_rope,
    rope,
    silu,
    scaled_dot_product_attention,
    cross_entropy,
)


def test_train_tiny_transformer():
    # Hyperparameters for a tiny model
    vocab_size = 100
    context_length = 8
    d_model = 16
    num_layers = 2
    num_heads = 2
    d_ff = 32
    rope_theta = 10000.0
    batch_size = 4
    num_epochs = 30
    lr = 1e-3

    # Random input indices
    in_indices = torch.randint(0, vocab_size, (batch_size, context_length))

    # Random weights for all layers
    weights = {}
    weights["token_embeddings.weight"] = torch.randn(vocab_size, d_model, requires_grad=True)
    for i in range(num_layers):
        prefix = f"layers.{i}."
        weights[prefix + "attn.q_proj.weight"] = torch.randn(d_model, d_model, requires_grad=True)
        weights[prefix + "attn.k_proj.weight"] = torch.randn(d_model, d_model, requires_grad=True)
        weights[prefix + "attn.v_proj.weight"] = torch.randn(d_model, d_model, requires_grad=True)
        weights[prefix + "attn.output_proj.weight"] = torch.randn(d_model, d_model, requires_grad=True)
        weights[prefix + "ln1.weight"] = torch.ones(d_model, requires_grad=True)
        weights[prefix + "ffn.w1.weight"] = torch.randn(d_ff, d_model, requires_grad=True)
        weights[prefix + "ffn.w2.weight"] = torch.randn(d_model, d_ff, requires_grad=True)
        weights[prefix + "ffn.w3.weight"] = torch.randn(d_ff, d_model, requires_grad=True)
        weights[prefix + "ln2.weight"] = torch.ones(d_model, requires_grad=True)
    weights["ln_final.weight"] = torch.ones(d_model, requires_grad=True)
    weights["lm_head.weight"] = torch.randn(vocab_size, d_model, requires_grad=True)

    # Collect all parameters for optimizer
    params = [w for w in weights.values() if w.requires_grad]
    print("params:", params)
    optimizer = torch.optim.Adam(params, lr=lr)
    # Before the loop
    targets = torch.randint(0, vocab_size, (batch_size, context_length))

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = transformer_lm(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights=weights,
            in_indices=in_indices,
        )
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        print(f"Epoch {epoch+1} | Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
    print("Training loop with multiple epochs completed successfully.")
