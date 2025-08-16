from datasets import load_dataset
from tqdm import tqdm
import io

dataset = load_dataset("Skylion007/openwebtext")['train']
split_dataset = dataset  # Use all data for training


with io.open('data/owt_train.txt','w') as fopen:
    listout = []
    for data in tqdm(split_dataset):
        listout.append(data['text']+'<|endoftext|>')
        if len(listout) > 1000:
            _ = fopen.write(''.join(listout))
            listout = []


with io.open('data/owt_valid.txt','w') as fopen:
    listout = []
    for data in tqdm(split_dataset):  # No separate test set
        listout.append(data['text']+'<|endoftext|>')
        if len(listout) > 1000:
            _ = fopen.write(''.join(listout))
            listout = []