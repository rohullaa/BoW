import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from dataset import Dataset
from model import MLP

from sklearn import metrics
from argparse import ArgumentParser

import numpy as np
import logging, tqdm 


def train():
    model.train()
    for feature_vector, label_true in tqdm.tqdm(train_iter):
        optimizer.zero_grad()
        label_pred = model(feature_vector)
        label_true = label_true.type(torch.LongTensor)

        loss = criterion(label_pred, label_true)
        loss.backward()
        optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, data_iter):
    model.eval()
    labels_true, predictions = [], []
    for feature_vector,label_true in tqdm.tqdm(data_iter):
        output = model(feature_vector)
        predictions += output.argmax(dim=1).tolist()
        labels_true += label_true.tolist()


    return metrics.f1_score(labels_true, predictions, average='macro')

def run_model():
    for epoch in range(args.epochs):
        _loss = train()
        scheduler.step()

        f1_train = evaluate(model, train_iter)
        f1_test = evaluate(model, val_iter)

        logger.info(f"Epoch: {epoch} - Train: {f1_train} - Test: {f1_test}")

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--input_size", action="store", type=int, default=100)
    parser.add_argument("--hidden_dim", action="store", type=int, default=50)
    parser.add_argument("--n_hidden_layers", action="store", type=int, default=3)
    parser.add_argument("--batch_size", action="store", type=int, default=32)
    parser.add_argument("--lr", action="store", type=float, default=1e-2)
    parser.add_argument("--epochs", action="store", type=int, default=60)
    parser.add_argument("--gamma", action="store", type=float, default=0.9)
    args = parser.parse_args()


    X_train = torch.load("data/X_train.pt")
    X_test = torch.load("data/X_test.pt")

    Y_train = np.loadtxt(fname = "data/Y_train.txt")
    Y_test = np.loadtxt(fname = "data/Y_test.txt")

    train_data = Dataset(X_train, Y_train)
    test_data = Dataset(X_test, Y_test)

    train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_iter = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = MLP(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    run_model()



