import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ProcessingModel(nn.Module):
    def __init__(self, input_channels=185, conv_out_channels=46, kernel_size=3,
                 n_conv_layers=2, transformer_layers=1, transformer_heads=1,
                 transformer_emb_size=185//4, dropout=0.1):
        super(ProcessingModel, self).__init__()

        self.conv_norm_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            conv_layer = nn.Conv1d(
                in_channels=input_channels if i == 0 else 185,
                out_channels=input_channels if i == 0 else 185,
                kernel_size=kernel_size, padding=kernel_size//2
            )
            pool = nn.MaxPool1d(3)
            norm_layer = nn.LayerNorm(33 if i == 0 else 11)
            self.conv_norm_layers.append(nn.Sequential(conv_layer, pool, norm_layer))

        self.pos_encoder = PositionalEncoding(d_model=transformer_emb_size,
                                              dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model=transformer_emb_size,
                                                 nhead=transformer_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=transformer_layers
        )

        self.linear_rectifier = nn.Linear(185, transformer_emb_size)
        self.linear = nn.Linear(transformer_emb_size, 39)

    def forward(self, src):
        src = torch.squeeze(src, dim=1)
        conv_output = src
        for conv_layer, pool, norm_layer in self.conv_norm_layers:
            conv_output = conv_layer(conv_output)
            conv_output = pool(conv_output)
            conv_output = norm_layer(conv_output)
            conv_output = F.relu(conv_output)

        conv_output = self.linear_rectifier(conv_output.contiguous().permute(0, 2, 1))

        # conv_output = self.pos_encoder(conv_output)
        transformer_output = self.transformer_encoder(conv_output.contiguous())
        transformer_output, _ = torch.max(transformer_output, dim=1)
        return self.linear(transformer_output)
