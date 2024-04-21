import re
import glob
import numpy as np
import torch
from torch.utils.data import Dataset,Subset,DataLoader
np.random.seed(42)
def load_book(address):
    with open(address,'r', errors='ignore',encoding='utf-8') as f:
        text = ''.join(f.readlines())
    return text

def preprocess_text(text):
    start_marker = '*** START OF THIS PROJECT GUTENBERG EBOOK'
    end_marker = '*** END OF THIS PROJECT GUTENBERG EBOOK'
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)

    if start_index != -1 and end_index != -1:
        text = text[start_index + len(start_marker):end_index].strip()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(' +', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text
    # Comment: How to handle end of text token?!?!?!?

def chunk_tokens(tokens, max_length=256):
    for i in range(0, len(tokens), max_length):
        yield tokens[i:i + max_length]


class CustomDataset(Dataset):
    def __init__(self, tokens_chunks,tokenizer,device):
        self.tokens_chunks = tokens_chunks[:-1]
        self.tokenizer=tokenizer
        self.device=device

    def __len__(self):
        return len(self.tokens_chunks)

    def __getitem__(self, idx):
        chunk_tokens=self.tokens_chunks[idx]
        ids=torch.tensor(self.tokenizer.convert_tokens_to_ids(chunk_tokens) )
        attention_mask=torch.ones(ids.shape)
        return {'input_ids':ids.to(self.device),'attention_mask':attention_mask.to(self.device)}

def pipeline_load_dataset(tokenizer,max_length=256,batch_size=4,device='cuda'):
    whole_text=''
    addresses=glob.glob('../data/*.txt')
    print(addresses)
    for address in addresses:
        text = load_book(address)
        processed_text = preprocess_text(text)
        whole_text=whole_text+" "+processed_text
    tokens = tokenizer.tokenize(whole_text)
    tokens_chunks = list(chunk_tokens(tokens,max_length))
    dataset=CustomDataset(tokens_chunks,tokenizer,device)
    indices=np.array([_ for _ in range(len(dataset))])
    train_flag=np.random.binomial(1, 0.7, indices.shape).astype('bool')
    train_indices=indices[train_flag]
    test_indices=indices[~train_flag]
    train_dataset=Subset(dataset,train_indices)
    test_dataset=Subset(dataset,test_indices)
    train_loader= DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader= DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_dataset,test_dataset,train_loader,test_loader


