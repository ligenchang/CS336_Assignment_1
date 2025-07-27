import os
import pickle
import torch
from cs336_basics.nn_utils import transformer_lm, cross_entropy
from cs336_basics.tokenizer import Tokenizer
import multiprocessing

tokenizer = None  # Will be loaded in main only

# Hyperparameters
vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
num_layers = 4
num_heads = 16
rope_theta = 10000
batch_size = 32
num_steps = 50000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
SPECIAL_TOKEN = "<|endoftext|>"
NUM_PROCESSES = 16  # Use 16 chunks for parallel pretokenization

def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(chunk):
    # Suppress all logs in worker processes
    import sys
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    import pickle
    from cs336_basics.tokenizer import Tokenizer
    with open("tinystories_bpe_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("tinystories_bpe_merges.pkl", "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    return list(tokenizer.encode_iterable(chunk.splitlines()))


def main():
    global tokenizer
    import time
    t0 = time.time()
    # Load BPE tokenizer
    with open("tinystories_bpe_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("tinystories_bpe_merges.pkl", "rb") as f:
        merges = pickle.load(f)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded BPE tokenizer with vocab size: {len(vocab)} (elapsed: {time.time()-t0:.2f}s)")
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    tokens_path = "tinystories_pretok_tokens.pkl"
    t1 = time.time()
    if os.path.exists(tokens_path):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading pre-tokenized tokens from {tokens_path} (elapsed: {time.time()-t1:.2f}s)...")
        with open(tokens_path, "rb") as f:
            tokens = pickle.load(f)
        num_tokens = len(tokens)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total tokens loaded: {num_tokens} (elapsed: {time.time()-t1:.2f}s)")
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finding chunk boundaries for parallel pretokenization...")
        t2 = time.time()
        with open(DATA_PATH, "rb") as f:
            boundaries = find_chunk_boundaries(f, NUM_PROCESSES, SPECIAL_TOKEN.encode("utf-8"))
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found chunk boundaries (elapsed: {time.time()-t2:.2f}s)")
            chunks = []
            t3 = time.time()
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Read chunk {start}-{end}, size: {len(chunk)} (elapsed: {time.time()-t3:.2f}s)")
                t3 = time.time()

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Tokenizing {len(chunks)} chunks in parallel...")
        t4 = time.time()
        with multiprocessing.Pool(NUM_PROCESSES) as pool:
            results = pool.map(pretokenize_chunk, chunks)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished tokenizing (elapsed: {time.time()-t4:.2f}s)")

        # Aggregate all tokens
        tokens = []
        t5 = time.time()
        for tks in results:
            tokens.extend(tks)
        num_tokens = len(tokens)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Total tokens: {num_tokens} (elapsed: {time.time()-t5:.2f}s)")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saving pre-tokenized tokens to {tokens_path}...")
        t6 = time.time()
        with open(tokens_path, "wb") as f:
            pickle.dump(tokens, f)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved tokens (elapsed: {time.time()-t6:.2f}s)")

    # Prepare batches
    t7 = time.time()
    def get_batch(tokens, batch_size, context_length):
        idx = torch.randint(0, len(tokens) - context_length - 1, (batch_size,))
        x = torch.stack([torch.tensor(tokens[i:i+context_length]) for i in idx])
        y = torch.stack([torch.tensor(tokens[i+1:i+context_length+1]) for i in idx])
        return x.to(device), y.to(device)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished preparing batches (elapsed: {time.time()-t7:.2f}s)")

    # Initialize weights dict for transformer_lm
    weights = {}
    weights["token_embeddings.weight"] = torch.randn(vocab_size, d_model, device=device) * 0.02
    for i in range(num_layers):
        prefix = f"layers.{i}."
        weights[prefix + "attn.q_proj.weight"] = torch.randn(d_model, d_model, device=device) * 0.02
        weights[prefix + "attn.k_proj.weight"] = torch.randn(d_model, d_model, device=device) * 0.02
        weights[prefix + "attn.v_proj.weight"] = torch.randn(d_model, d_model, device=device) * 0.02
        weights[prefix + "attn.output_proj.weight"] = torch.randn(d_model, d_model, device=device) * 0.02
        weights[prefix + "ln1.weight"] = torch.ones(d_model, device=device)
        weights[prefix + "ln2.weight"] = torch.ones(d_model, device=device)
        weights[prefix + "ffn.w1.weight"] = torch.randn(d_ff, d_model, device=device) * 0.02
        weights[prefix + "ffn.w2.weight"] = torch.randn(d_model, d_ff, device=device) * 0.02
        weights[prefix + "ffn.w3.weight"] = torch.randn(d_ff, d_model, device=device) * 0.02
    weights["ln_final.weight"] = torch.ones(d_model, device=device)
    weights["lm_head.weight"] = torch.randn(vocab_size, d_model, device=device) * 0.02
    # Optimizer and scheduler
    from cs336_basics.optimizer import AdamW
    optimizer = AdamW(weights.values(), lr=5e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    # Training loop
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting training...")
    t8 = time.time()
    best_loss = float('inf')
    for step in range(num_steps):
        x, y = get_batch(tokens, batch_size, context_length)
        logits = transformer_lm(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights=weights,
            in_indices=x
        )
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(weights, "tinystories_transformer_best.pt")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Step {step}: New best loss {best_loss:.4f}, model saved.")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Step {step}: loss={loss.item():.4f}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished training (elapsed: {time.time()-t8:.2f}s)")

    # Save model checkpoint
    # torch.save(weights, "tinystories_transformer.pt")

    # --- Text Generation ---
    def sample(transformer_lm, weights, tokenizer, context, max_tokens=256, temperature=1.0, top_p=0.95):
        generated = context[:]
        for _ in range(max_tokens):
            input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = transformer_lm(
                    vocab_size=vocab_size,
                    context_length=input_ids.shape[1],
                    d_model=d_model,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_theta=rope_theta,
                    weights=weights,
                    in_indices=input_ids
                )
                next_logits = logits[0, -1, :] / temperature
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cumulative_probs > top_p
                if torch.any(cutoff):
                    cutoff_idx = torch.where(cutoff)[0][0]
                    sorted_logits = sorted_logits[:cutoff_idx+1]
                    sorted_indices = sorted_indices[:cutoff_idx+1]
                probs = torch.softmax(sorted_logits, dim=-1)
                next_token = sorted_indices[torch.multinomial(probs, 1).item()].item()
            generated.append(next_token)
            if next_token == tokenizer.byte_to_id.get(b'<|endoftext|>'):
                break
        return generated

    # Improved prompt and decoding parameters for fluency
    prompt = (
        "Once upon a time, there was a little girl named Lily who loved adventures. "
        "One day, she found a mysterious map in her grandmother's attic. "
        "The map promised to lead to a hidden treasure, but only if Lily could solve its riddles. "
        "Excited, Lily packed her backpack and set off on her journey. "
    )
    context = tokenizer.encode(prompt)
    output_ids = sample(transformer_lm, weights, tokenizer, context, max_tokens=256, temperature=0.7, top_p=0.9)
    output_text = tokenizer.decode(output_ids)
    print("\n--- Generated Text ---\n")
    print(output_text)
    print("\n--- End Generated Text ---\n")
    print("# The fluency of the output depends on (1) the amount and quality of training, (2) decoding parameters like temperature and top-p, (3) the prompt/context used for generation, and (4) repetition penalty.")

if __name__ == "__main__":
    main()