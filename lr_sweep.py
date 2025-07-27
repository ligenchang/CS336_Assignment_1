import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from cs336_basics.tokenizer import Tokenizer

# --- Hyperparameters ---
vocab_size = 10000
context_length = 256
d_model = 512
d_ff = 1344
num_layers = 4
num_heads = 16
batch_size = 32
num_steps = 1000  # For sweep, use fewer steps for speed
learning_rates = [5e-5, 1e-4, 2e-4, 3e-4, 5e-4]

DATA_PATH = "data/TinyStoriesV2-GPT4-train.txt"
SPECIAL_TOKEN = "<|endoftext|>"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Load Tokenizer ---
with open("tinystories_bpe_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("tinystories_bpe_merges.pkl", "rb") as f:
    merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

# --- Load Pre-tokenized Tokens ---
tokens_path = "tinystories_pretok_tokens.pkl"
with open(tokens_path, "rb") as f:
    tokens = pickle.load(f)

# --- Validation Split ---
val_frac = 0.05
val_size = int(len(tokens) * val_frac)
train_tokens = tokens[:-val_size]
val_tokens = tokens[-val_size:]

# --- Model Definition ---
class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, num_heads, context_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits

# --- Batch Functions ---
def get_batch(tokens, batch_size, context_length):
    idx = torch.randint(0, len(tokens) - context_length - 1, (batch_size,))
    x = torch.stack([torch.tensor(tokens[i:i+context_length]) for i in idx])
    y = torch.stack([torch.tensor(tokens[i+1:i+context_length+1]) for i in idx])
    return x.to(device), y.to(device)

def eval_loss(model, tokens, batch_size, context_length, num_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(tokens, batch_size, context_length)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
            losses.append(loss.item())
    return sum(losses) / len(losses)

# --- Sweep ---
results = {}
for lr in learning_rates:
    print(f"\n--- Training with learning rate {lr} ---")
    model = SimpleTransformerLM(vocab_size, d_model, d_ff, num_layers, num_heads, context_length).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    train_curve = []
    val_curve = []
    best_val_loss = float('inf')
    for step in range(num_steps):
        model.train()
        x, y = get_batch(train_tokens, batch_size, context_length)
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0:
            val_loss = eval_loss(model, val_tokens, batch_size, context_length)
            train_curve.append(loss.item())
            val_curve.append(val_loss)
            print(f"Step {step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"tinystories_transformer_best_lr{lr}.pt")
    results[lr] = (train_curve, val_curve)
    print(f"Best validation loss for lr={lr}: {best_val_loss:.4f}")

# --- Plot Learning Curves ---
plt.figure(figsize=(10,6))
for lr in learning_rates:
    train_curve, val_curve = results[lr]
    steps = [i*100 for i in range(len(train_curve))]
    plt.plot(steps, val_curve, label=f"val lr={lr}")
plt.xlabel("Step")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs. Step for Different Learning Rates")
plt.legend()
plt.grid()
plt.savefig("learning_rate_sweep.png")
plt.show()

print("Sweep complete. See 'learning_rate_sweep.png' for curves.")
