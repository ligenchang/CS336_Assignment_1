from datasets import load_dataset
from tqdm import tqdm
import io

ds = load_dataset("wikimedia/wikipedia", "20231101.en")




with io.open('data/wikipedia_train.txt','w') as fopen:
    listout = []
    for data in tqdm(ds['train']):
        listout.append(data['text']+'<|endoftext|>')
        if len(listout) > 1000:
            _ = fopen.write(''.join(listout))
            listout = []

