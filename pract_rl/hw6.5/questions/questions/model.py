import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class FixedEmbeddingsModel(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(FixedEmbeddingsModel, self).__init__()
        self.embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.rnn = nn.RNN(input_size=embeddings.shape[1], hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, embeddings.shape[1])

    def forward(self, x, h=None):
        in_seq = self.embeddings(x)
        rnn_out, rnn_hidden = self.rnn(in_seq, h)
        out = self.out(rnn_out.data)
        return out, rnn_hidden


