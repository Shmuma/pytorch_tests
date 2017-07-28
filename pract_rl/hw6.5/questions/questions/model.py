import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class FixedEmbeddingsModel(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(FixedEmbeddingsModel, self).__init__()

        # create fixed embeddings layer
        self.embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        self.embeddings.weight.requires_grad = False

        self.rnn = nn.RNN(input_size=embeddings.shape[1], hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, embeddings.shape[0])

    def forward(self, x, h=None):

        if isinstance(x, rnn_utils.PackedSequence):
            x_emb = self.embeddings(x.data)
            rnn_in = rnn_utils.PackedSequence(data=x_emb, batch_sizes=x.batch_sizes)
        else:
            rnn_in = self.embeddings(x)
        rnn_out, rnn_hidden = self.rnn(rnn_in, h)
        out = self.out(rnn_out.data)
        return out, rnn_hidden


