# Bag of Words (BoW) Document Classification

This repo constains a simple bag-of-words document classifier. The bag-of-words model is a way of representing text data when modeling text with machine learning algorithms. The machine learning algorithms used is a simple feed-forward neural network. 

## Run the code
1. Produce the BOW representations
```
python3 dataset.py --vocab_size VOCAB_SIZE
```

2. Run the model
```
python3 main.py --input_size INPUT_SIZE
                --hidden_dim HIDDEN_DIM
                --n_hidden_layers N_HIDDEN_LAYERS
                --batch_size BATCH_SIZE
                --lr LR
                --epochs EPOCHS
                --gamma GAMMA
```



## Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch ```
4. tqdm ```pip install tqdm```