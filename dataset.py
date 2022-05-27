from typing import List
import torch
from collections import Counter
from pandas import DataFrame


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: DataFrame, vocab_size: int):
        texts = list(data['text'].str.split(" "))
        sources = list(data['source'])

        tokens = Counter([token for text in texts for token in text])
        text_vocab = [token for token, freq in tokens.most_common(vocab_size)]
        label_vocab = list(sorted(set(sources)))
        self.vocab = {'text': text_vocab, 'label': label_vocab}

        self.num_features = vocab_size
        self.num_labels = len(self.vocab['label'])

        text_indexer = {token: i for i, token in enumerate(self.vocab['text'])}
        label_indexer = {token: i for i, token in enumerate(self.vocab['label'])}
        self.indexers = {'text': text_indexer, 'label': label_indexer}

        self.bow_vectors = [self.create_bow(text) for text in texts]
        self.labels = [self.indexers['label'][source] for source in sources]

    def create_bow(self, text: List[str]):
        bow = torch.zeros(self.num_features, dtype=torch.float)
        for token in text:
            if token in self.vocab['text']:
                bow[self.indexers['text'][token]] += 1
        return bow

    def __getitem__(self, index: int):
        return self.bow_vectors[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    pass
