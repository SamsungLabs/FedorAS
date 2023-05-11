# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

from functools import partial as p

import torch
import torch.nn as nn
import torch.nn.modules.rnn as rnn

import src.models.spos as spos
from src.models.rnn_ops import LiGRU, QuasiRNN, Identity, Zero, Linear, Conv


class _GenericBlock(nn.Module):
    def __init__(self, dim, in_dim=None, inc_identity=True, inc_zero=True, is_reduction=False):
        super().__init__()

        if in_dim is None:
            in_dim = dim

        extra = []
        if inc_identity:
            extra.append(p(Identity))
        if inc_zero:
            extra.append(p(Zero, output_size=(None if not is_reduction else dim)))

        self.op = spos.SelectOp(
            p(rnn.LSTM, in_dim, dim, batch_first=True),
            p(rnn.GRU, in_dim, dim, batch_first=True),
            p(LiGRU, in_dim, dim, batch_size=4), # batch_size here doesn't really matter, it's only about performance
            p(QuasiRNN, in_dim, dim),
            p(Linear, in_dim, dim),
            p(Conv, in_dim, dim),
            *extra
        )

    def forward(self, x):
        res = self.op(x)

        if not(isinstance(res, tuple)):
            return res, None
        return res


class StemBlock(_GenericBlock):
    def __init__(self, dim, in_dim=None):
        super().__init__(dim, in_dim, inc_zero=False, inc_identity=False)

class BranchBlock(_GenericBlock):
    def __init__(self, dim, in_dim=None):
        super().__init__(dim, in_dim, inc_zero=True, inc_identity=False, is_reduction=True)

class Block(_GenericBlock):
    def __init__(self, dim, in_dim=None):
        super().__init__(dim, in_dim, inc_zero=False, inc_identity=True)

class ScBlock(_GenericBlock):
    def __init__(self, dim, in_dim=None):
        super().__init__(dim, in_dim, inc_zero=True, inc_identity=True, is_reduction=False)


class ShakespeareModel(nn.Module):
    def __init__(self, *args, vocab_size=90, embedding_dim=8, hidden_dim=128, **kwargs):
        kwargs.pop('num_classes', None)
        super().__init__()

        # set class variables
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = StemBlock(hidden_dim, embedding_dim)
        self.branch = BranchBlock(hidden_dim, embedding_dim)
        self.lstm2 = Block(hidden_dim)
        self.sc = ScBlock(hidden_dim)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = x.to(dtype=torch.int32).clamp(0, self.vocab_size-1)
        x = self.embedding(x)

        y1, _ = self.lstm1(x)

        if not isinstance(self.branch, Zero):
            y2, _ = self.branch(x)
            y = y1 + y2
        else:
            y = y1

        z1, _ = self.lstm2(y)

        if not isinstance(self.sc, Zero):
            z2, _ = self.sc(y)
            z = z1 + z2
        else:
            z = z1

        out = self.fc(z)
        out = out.reshape(-1, self.vocab_size)
        return out

    def archive(self):
        if isinstance(self, spos.SPOSMixin):
            return None
        return super().archive()


class ShakespeareFjord(nn.Module):
    def __init__(self, vocab_size=90, embedding_dim=8, hidden_dim=128):
        super().__init__()

        # set class variables
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # embedding layer is not prune, register lstm1 as first layer
        self.lstm1.is_first = True
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = [None, None]

        input = input.to(dtype=torch.int32).clamp(0, self.vocab_size-1)
        embeds = self.embedding(input)
        lstm_out, hidden_1 = self.lstm1(embeds, hidden[0])
        lstm_out, hidden_2 = self.lstm2(lstm_out, hidden[1])
        out = self.fc(lstm_out)
        # flatten the output
        out = out.reshape(-1, self.vocab_size)
        return out, [hidden_1, hidden_2]

    def init_hidden(self, batch_size, device):
        hidden = [(torch.zeros(1, batch_size, self.hidden_dim).to(device),
                   torch.zeros(1, batch_size, self.hidden_dim).to(device)),
                  (torch.zeros(1, batch_size, self.hidden_dim).to(device),
                   torch.zeros(1, batch_size, self.hidden_dim).to(device))]
        return hidden
