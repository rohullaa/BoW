import torch
import pandas as pd
import numpy as np
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_path, vocab_size):
        self.vocab_size = vocab_size
        self.text, self.source = self.load_data(data_path)
        self.vocab = self.make_vocab(self.text, self.source, self.vocab_size)
        self.text_indexer = {token: i for i, token in enumerate(self.vocab['text'])}


        self.y = self.convert_to_ints(self.source)
        self.X = [self.make_bow(t) for t in self.text]



    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def make_vocab(self,texts,sources, vocab_size):
        tokens = Counter([token for text in texts for token in text])
        text_vocab = [token for token, freq in tokens.most_common(vocab_size)]
        label_vocab = list(sorted(set(sources)))
        vocab = {'text': text_vocab, 'label': label_vocab}

        return vocab
    
    def load_data(self, data_path):
        df = pd.read_csv(
            filepath_or_buffer = data_path,
            sep = "\t"
        )
        text = df.text
        source = df.source

        return text, source

    def make_bow(self, text):
        bow = torch.zeros(self.vocab_size, dtype=torch.float)
        for token in text:
            if token in self.vocab['text']:
                bow[self.text_indexer[token]] += 1
        return bow
        
    
    def split(self):
        pass

    def convert_to_ints(self, source):
        unique_labels = sorted(set(list(source)))
        self.label2idx = {label:i for i,label in enumerate(unique_labels)}
        y_new = np.array([self.label2idx[i] for i in list(source)])
        return y_new



if __name__ == "__main__":
    data = Dataset(data_path = "data/signal_20_obligatory1_train.tsv.gz",vocab_size= 100)
    
    for i,j in zip(data.X, data.text):
        print(i,j)