import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = (torch.matmul(Q, K.transpose(-2, -1)) /
                            torch.sqrt(torch.tensor(K.size(-1),
                                                    dtype=torch.float32)))
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)
        return attention_output


class AttentionRNN(nn.Module):
    def __init__(self, input_dim, attention_dim, rnn_hidden_size, num_classes):
        super(AttentionRNN, self).__init__()

        self.attention = Attention(input_dim, attention_dim)

        self.rnn = nn.GRU(input_size=attention_dim,
                          hidden_size=rnn_hidden_size,
                          num_layers=1, batch_first=True)

        # Couche de sortie
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):

        x = self.attention(x)

        x, _ = self.rnn(x)

        x = x[:, -1, :]
        x = self.fc(x)

        return x
