from typing import List
import torch
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, vocab_size: int, vocab:List = None):
        texts = list(data['text'].str.split(" "))
        sources = list(data['source'])
        
        tokens = Counter([token for text in texts for token in text])
        text_vocab = [token for token, freq in tokens.most_common(vocab_size)]

        label_vocab = list(sorted(set(sources)))
        logger.info("Loaded the data")
        
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {'text': text_vocab, 'label': label_vocab}
        logger.info("Loaded the vocab")

        self.num_features = vocab_size
        self.num_labels = len(self.vocab['label'])

        text_indexer = {token: i for i, token in enumerate(self.vocab['text'])}
        label_indexer = {token: i for i, token in enumerate(self.vocab['label'])}
        self.indexers = {'text': text_indexer, 'label': label_indexer}

        self.bow_vectors = [self.create_bow(text) for text in texts]
        self.labels = [self.indexers['label'][source] for source in sources]
        logger.info("Loaded the BOW vectors")

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

class Split():
    def __init__(self, vocab_size):
        self.df = pd.read_csv(
        filepath_or_buffer = "data/signal_20_obligatory1_train.tsv.gz", 
        sep = "\t"
        )
        train, test = train_test_split(self.df,
            test_size=0.20,
            stratify = self.df.source,
            random_state = 5550
        )

        logger.info("TRAINING DATA:")
        self.train_dataset = Dataset(train, vocab_size)

        logger.info("TEST DATA:")
        self.test_dataset = Dataset(test, vocab_size, self.train_dataset.vocab)



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = Split(vocab_size=100)
    
    train_dataset = split.train_dataset
    test_dataset = split.test_dataset   






