import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        pos = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
    
        pe[:, :, 0::2] = torch.sin(pos * div_term)
        pe[:, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, n_tokens, d_model, n_head, n_hidden, n_layers, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, n_hidden, dropout_prob, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.decoder = nn.Linear(d_model, n_tokens)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x, mask)
        x = self.decoder(x)
        return x

    def init_weights(self):
        initrange = 0.1
        self.embedding.weights.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zeros_()
        self.decover.weights.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)