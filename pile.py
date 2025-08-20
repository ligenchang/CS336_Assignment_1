import datasets

def save_subset_as_text(output_file, num_samples=10000):
    # Stream The Pile from Hugging Face (no The-Eye downloads)
    dataset = datasets.load_dataset(
        "EleutherAI/pile",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    # Take first `num_samples`
    iterator = iter(dataset)
    count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for example in iterator:
            if "text" in example:
                f.write(example["text"].strip() + "\n\n")
            count += 1
            if count >= num_samples:
                break

    print(f"✅ Saved {count} samples to {output_file}")


if __name__ == "__main__":
    save_subset_as_text("pile_subset.txt", num_samples=10000)
