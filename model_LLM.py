import torch.nn as nn
import torch


# globalement ça marche pas ce qui n'est pas décoddant, voir les megs
# comme des embeddings c'est pas forcément naturel. Il faudrait réduite
# les megs avant ???

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(self._get_positional_encoding(max_len,
                                                                   d_model),
                                     requires_grad=False)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len]

    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)


class Model(nn.Module):
    def __init__(self, n_layers, num_heads=5):
        super(Model, self).__init__()
        self.n_layers = n_layers
        self.positional_encoding = PositionalEncoding(50, 306)

        Layer = nn.TransformerEncoderLayer(d_model=50, nhead=num_heads,
                                           dim_feedforward=200,
                                           dropout=0.1,
                                           activation='gelu',
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=Layer,
                                             num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.linear = nn.Linear(306, 39)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=2)

        return self.linear(x)
