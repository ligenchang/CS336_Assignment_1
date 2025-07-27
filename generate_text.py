import torch
import pickle
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Model Definition (must match training) ---
class SimpleTransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, num_heads, context_length):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_ff, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits

# --- Load Model Checkpoint ---
model = SimpleTransformerLM(vocab_size, d_model, d_ff, num_layers, num_heads, context_length).to(device)
model.load_state_dict(torch.load("tinystories_transformer_best.pt", map_location=device))
model.eval()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loaded model checkpoint.")

# --- Text Generation Function ---
def sample(model, tokenizer, context, max_tokens=256, temperature=0.7, top_p=0.9):
    model.eval()
    generated = context[:]
    input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_tokens):
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
        input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
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
