
import torch
import pickle
from cs336_basics.nn_utils import TransformerLM
from cs336_basics.tokenizer import Tokenizer
import time

# --- Load Tokenizer ---
with open("tinystories_bpe_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("tinystories_bpe_merges.pkl", "rb") as f:
    merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# --- Model Hyperparameters (must match training) ---
vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
num_layers = 4
num_heads = 16
rope_theta = 10000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load Model Weights ---
weights = torch.load("tinystories_transformer_ckpt.pt", map_location=device)
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded model weights.")

# --- Create Model Instance ---
model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
    device=device,
    dtype=torch.float32
)
model.load_state_dict(weights)
model.eval()

# --- Text Generation Function ---
def sample(model, tokenizer, context, max_tokens=256, temperature=0.7, top_p=0.9):
    generated = context[:]
    for _ in range(max_tokens):
        input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)
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

# --- Example Usage ---
prompt = "Once upon a time, there was a little girl named Lily who loved adventures. One day,"
context = tokenizer.encode(prompt)
output_ids = sample(model, tokenizer, context, max_tokens=256, temperature=1.0, top_p=0.95)
output_text = tokenizer.decode(output_ids)
print("\n--- Generated Text ---\n")
print(output_text)
print("\n--- End Generated Text ---\n")
print("# The fluency of the output depends on (1) the amount and quality of training, (2) decoding parameters like temperature and top-p, and (3) the prompt/context used for generation.")
