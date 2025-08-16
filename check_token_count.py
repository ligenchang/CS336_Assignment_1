import pickle

tokens_path = "wikipedia_pretok_tokens.pkl"

total_tokens = 0
total_docs = 0

with open(tokens_path, "rb") as f:
    while True:
        try:
            obj = pickle.load(f)
        except EOFError:
            break
        if isinstance(obj, (list, tuple)):
            total_tokens += len(obj)
            total_docs += 1
        elif hasattr(obj, 'shape') and hasattr(obj, '__len__'):
            total_tokens += len(obj)
            total_docs += 1

print(f"[INFO] Total tokens: {total_tokens:,}")
print(f"[INFO] Total documents: {total_docs:,}")
