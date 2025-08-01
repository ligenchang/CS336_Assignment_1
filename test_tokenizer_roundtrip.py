import sys
from cs336_basics.tokenizer import Tokenizer
'''
python test_tokenizer_roundtrip.py --vocab owt_bpe_vocab.pkl --merges owt_bpe_merges.pkl --text "Your test string here"
'''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--merges', type=str, required=True)
    parser.add_argument('--text', type=str, required=True, help='Text to test round-trip')
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges)
    encoded = tokenizer.encode(args.text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {args.text}")
    print(f"Encoded token IDs: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip match: {args.text == decoded}")

    input_bytes = len(args.text.encode('utf-8'))
    num_tokens = len(encoded)
    if num_tokens > 0:
        compression_ratio = input_bytes / num_tokens
    else:
        compression_ratio = float('inf')
    print(f"\nInput bytes: {input_bytes}, Number of tokens: {num_tokens}, Compression ratio (bytes/token): {compression_ratio:.3f}")

    print("\nTokenization mapping:")
    for token_id in encoded:
        token_bytes = tokenizer.vocab[token_id]
        try:
            token_str = token_bytes.decode('utf-8', errors='replace')
        except Exception:
            token_str = str(token_bytes)
        print(f"Token: {repr(token_str)}\tID: {token_id}")
