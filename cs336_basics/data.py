import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Calculate how many valid starting positions we have in the dataset
    max_start_idx = len(dataset) - context_length
    
    # Randomly sample batch_size starting indices
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    # Create input sequences and corresponding labels
    x = np.zeros((batch_size, context_length), dtype=np.int64)
    y = np.zeros((batch_size, context_length), dtype=np.int64)
    
    # Fill in the sequences and labels
    for i, start_idx in enumerate(start_indices):
        # Input sequence is a slice of the dataset starting from start_idx
        x[i, :] = dataset[start_idx:start_idx + context_length]
        # Labels are the input sequence shifted by 1 (next token prediction)
        y[i, :] = dataset[start_idx + 1:start_idx + context_length + 1]
    
    # Convert to PyTorch tensors and move to the specified device
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    
    return x_tensor, y_tensor
