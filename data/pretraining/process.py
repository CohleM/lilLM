import numpy as np
import os
import argparse

from transformers import AutoTokenizer
from datasets import load_dataset # huggingface dataset
from tqdm import tqdm

if __name__=='__main__':
    #file_path = '/Users/cohlem/Projects/Experimentation/lillm/model/tokenizer/'
    parser = argparse.ArgumentParser(description='Loading the tokenizer')
    parser.add_argument('--tokenizer_path', required=True, type=str, help='Path to tokenizer')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
#    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    num_proc = 50 

    dataset = load_dataset('text', num_proc=num_proc,data_files = 'input.txt') #for loading custom data
    #dataset = load_dataset("Skylion007/openwebtext", num_proc=num_proc) #for loading custom data
    split_dataset = dataset['train'].train_test_split(test_size = 0.0010, shuffle=True, seed=43)
    split_dataset['val'] = split_dataset.pop('test')

    def tokenize(item):
        ids = tokenizer.encode(item['text'] + '<|endoftext|>')
        return {'ids': ids, 'len': len(ids)}

    #under the hood, data is broken down into shards/batches, accessed using Memory mapping, and only processing batches in the RAM. 
    #See https://huggingface.co/docs/datasets/v2.1.0/en/about_arrow#:~:text=Memory%2Dmapping,with%20relatively%20small%20device%20memory.
    # essentially not loaded entirely in RAM but via memory mapping loads only what needs to be processed
    tokenized_data = split_dataset.map(tokenize, remove_columns='text', num_proc=num_proc)


        # Writing .bin file but processing in batches
    for split, dset in tokenized_data.items():
        batch_size = 1024
        idx = 0
        arr_sz = np.sum(dset['len'])

        # Memory mapping of file to our array
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_sz,)) 

        for b in tqdm(range(batch_size), desc=f'processing {filename}'):
            shard = dset.shard(num_shards=batch_size, index=b, contiguous=True).with_format('numpy')
            shard = np.concatenate(shard['ids'])

            arr[idx: idx + len(shard)] = shard # write the shard to virtual memory page cache (in RAM), OS writes to the file whenever it feels necessary
            idx += len(shard)

        arr.flush() # force OS to clear the page cache and write to the disk



