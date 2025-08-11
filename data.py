from datasets import load_dataset
from tqdm import tqdm
import io

dataset = load_dataset("Skylion007/openwebtext")['train']
split_dataset = dataset.train_test_split(train_size=2400000, test_size=60000, seed=0)


with io.open('data/owt_train.txt','w') as fopen:
    listout = []
    for data in tqdm(split_dataset['train']):
        listout.append(data['text']+'<|endoftext|>')
        if len(listout) > 1000:
            _ = fopen.write(''.join(listout))
            listout = []


with io.open('data/owt_valid.txt','w') as fopen:
    listout = []
    for data in tqdm(split_dataset['test']):
        listout.append(data['text']+'<|endoftext|>')
        if len(listout) > 1000:
            _ = fopen.write(''.join(listout))
            listout = []