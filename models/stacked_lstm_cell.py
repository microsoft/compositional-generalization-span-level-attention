""" Implementation of ONMT RNN for Input Feeding Decoding """
from typing import Tuple

import torch
import torch.nn as nn


class StackedLSTMCell(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super(StackedLSTMCell, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, x: torch.Tensor, h_tm1: Tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            x: shape (batch_size, input_size)
            h_tm1: shape (batch_size, layer_num, hidden_size)
        """
        h_0, c_0 = h_tm1
        h_0, c_0 = h_0.permute(1, 0, -1), c_0.permute(1, 0, -1)
        h_1, c_1 = [], []

        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(x, (h_0[i], c_0[i]))
            x = h_1_i

            if i + 1 != self.num_layers:
                x = self.dropout(x)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1, dim=1)
        c_1 = torch.stack(c_1, dim=1)

        return x, (h_1, c_1)
