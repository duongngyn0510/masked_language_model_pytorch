from dataset import MyDataset
from train import train
from predict import predict
from model import TransformerModel
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np



def data_collate_fn(batch_samples_list):
    arr = np.array(batch_samples_list)
    inputs = tokenizer(arr.tolist(), padding='max_length', max_length=30, return_tensors='pt')
    return inputs

SRC_DIR = './data.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenization=True)

dataset = MyDataset(SRC_DIR, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collate_fn)

n_tokens = tokenizer.vocab_size # the size of vocabulary
d_model = 200 # embedding dimension
n_hidden = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
n_head = 2 # the number of heads in the multiheadattention models
dropout_prob = 0.1 # the dropout value
model = TransformerModel(n_tokens, d_model, n_head, n_hidden, n_layers, dropout_prob).to(device)

epochs = 400
criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=0.0001)


if __name__ == '__main__':
    train(model, n_tokens, criterion, optim, dataloader, epochs, device)

    # Predict
    print('Testing: \n')
    with open(SRC_DIR, 'r') as f:
        text = f.readlines()
    print("Input: {}".format(text[0].strip()))
    pred_inp = tokenizer("Don't speak ill of [MASK].", return_tensors='pt')
    out = predict(model, pred_inp['input_ids'], device)
    print("Output: {}\n".format(tokenizer.decode(out)))