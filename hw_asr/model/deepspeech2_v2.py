import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

from math import floor


class RNN_with_BN(nn.Module):
    def __init__(self, str_rnn_type, rnn_type, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.bn = nn.BatchNorm1d(self.input_size)
        if str_rnn_type == 'RNN':
            self.rnn = rnn_type(self.input_size, self.hidden_size,
                                num_layers=self.num_layers,
                                nonlinearity='relu',
                                bias=True, batch_first=True,
                                bidirectional=self.bidirectional)
        else:
            self.rnn = rnn_type(self.input_size, self.hidden_size,
                                num_layers=self.num_layers,
                                bias=True, batch_first=True,
                                bidirectional=self.bidirectional)

    def forward(self, x, lengths):
        y = self.bn(x.transpose(1, 2)).transpose(1, 2)
        y = nn.utils.rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
        y, _ = self.rnn(y)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        if self.bidirectional:
            y = y.reshape(y.shape[0], y.shape[1], 2, self.hidden_size).sum(dim=2)
        return y


class LookaheadConv_with_BN_and_ReLU(nn.Module):
    def __init__(self, lookahead_ts, hidden_size):
        super().__init__()
        self.lookahead_ts = lookahead_ts
        self.hidden_size = hidden_size

        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, self.lookahead_ts,
                              padding=0, groups=self.hidden_size, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = nn.functional.pad(x, (0, self.lookahead_ts - 1), mode='constant', value=0.0)
        y = self.conv(y)
        y = self.bn(y)
        y = self.relu(y)
        return y


class DeepSpeech2_v2(BaseModel):
    def __init__(self, n_feats, n_class,
                 n_rnns, rnn_type, num_layers, bidirectional, hidden_size,
                 **batch):
        super().__init__(n_feats, n_class, **batch)
        self.n_feats = n_feats
        self.n_class = n_class

        # convolutions
        # B x n_mels x time -> B x 1 x n_mels x time
        self.n_convs = 3
        self.convs = Sequential(
            nn.Conv2d(1, 32, (41, 11), stride=(2, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, (21, 11), stride=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 96, (21, 11), stride=(2, 1), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        # B x 32 x new_n_mels x new_time

        # rnns
        # B x 32 x new_n_mels x new_time -> B x new_time x 32 * new_n_mels
        self.n_rnns = n_rnns
        assert rnn_type in ['RNN', 'LSTM', 'GRU']
        if rnn_type == 'RNN':
            self.rnn_type = nn.RNN
        elif rnn_type == 'LSTM':
            self.rnn_type = nn.LSTM
        else:
            self.rnn_type = nn.GRU
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_input_size = floor((self.n_feats - (41 - 1) - 1) / 2 + 1)
        self.rnn_input_size = floor((self.rnn_input_size - (21 - 1) - 1) / 2 + 1)
        self.rnn_input_size = floor((self.rnn_input_size - (21 - 1) - 1) / 2 + 1)
        self.rnn_input_size *= 96
        self.hidden_size = hidden_size
        self.rnns = Sequential(
            RNN_with_BN(rnn_type, self.rnn_type, self.rnn_input_size, self.hidden_size,
                        num_layers=self.num_layers, bidirectional=self.bidirectional),
            *(
                RNN_with_BN(rnn_type, self.rnn_type, self.hidden_size, self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)
                for _ in range(self.n_rnns - 1)
            )
        )
        # B x new_time x hidden_size

        # lookahead convolution
        # B x new_time x hidden_size -> B x hidden_size x new_time
        if not self.bidirectional:
            self.lookahead_ts = 80
            self.lookahead_conv = LookaheadConv_with_BN_and_ReLU(self.lookahead_ts,
                                                                 self.hidden_size)
        # B x hidden_size x new_time

        # fully connected layer
        # B x hidden_size x new_time -> B x new_time x hidden_size
        self.fc = torch.nn.Linear(self.hidden_size, self.n_class)
        # B x new_time x n_class

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.unsqueeze(1)
        x = self.convs(x)

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        new_spec_length = self.transform_input_lengths(spectrogram_length)
        for rnn in self.rnns._modules.values(): 
            x = rnn(x, new_spec_length)

        if not self.bidirectional:
            x = x.transpose(1, 2)
            x = self.lookahead_conv(x)
            x = x.transpose(1, 2)

        return {"logits": self.fc(x)}

    def transform_input_lengths(self, input_lengths):
        new_time = torch.floor((input_lengths - (11 - 1) - 1) / 2 + 1)
        new_time = torch.floor((new_time - (11 - 1) - 1) / 1 + 1)
        new_time = torch.floor((new_time - (11 - 1) - 1) / 1 + 1)
        return new_time.type(torch.int)   
