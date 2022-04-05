from typing import Tuple
from overrides import overrides

import torch

from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from torch.nn.utils.rnn import pad_packed_sequence


@Seq2SeqEncoder.register("stacked_lstm")
class StackedLstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module, stateful=False)

    @overrides
    def forward(
        self, inputs: torch.Tensor, mask: torch.BoolTensor = None, hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, total_sequence_length = inputs.shape[:2]

        if mask is None:
            mask = inputs.new_full(batch_size, total_sequence_length, fill_value=True, dtype=torch.bool)

        packed_sequence_output, final_states, restoration_indices = self.sort_and_run_forward(
            self._module, inputs, mask, hidden_state
        )
        # (batch_size, num_layers * num_directions, hidden_size)
        last_state, last_cell = final_states[0].permute(1, 0, 2), final_states[1].permute(1, 0, 2)
        num_directions = 2 if self._module.bidirectional else 1
        # (batch_size, num_layers, num_directions, hidden_size)
        last_state = last_state.view(batch_size, self._module.num_layers, num_directions, -1)
        last_cell = last_cell.view(batch_size, self._module.num_layers, num_directions, -1)

        # Shape: (non_empty_seq_num, total_sequence_length, output_size)
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)

        num_valid = unpacked_sequence_tensor.size(0)

        # Add back invalid rows.
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)

            hidden_state_zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, output_dim)
            last_state = torch.cat([last_state, hidden_state_zeros], dim=0)
            last_cell = torch.cat([last_cell, hidden_state_zeros], dim=0)

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2SeqEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.new_zeros(
                batch_size, sequence_length_difference, unpacked_sequence_tensor.size(-1)
            )
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        # Restore the original indices and return the sequence.
        output = unpacked_sequence_tensor.index_select(0, restoration_indices)
        # (batch_size, num_layers, num_directions, hidden_size)
        last_state = last_state.index_select(0, restoration_indices)
        last_cell = last_cell.index_select(0, restoration_indices)

        return output, (last_state, last_cell)


def test():
    lstm = StackedLstmSeq2SeqEncoder(3, 2, 2)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
    x = torch.randn(3, 5, 3)

    y, (h_t, c_t) = lstm(x, mask)
    assert (y[0, 2] - h_t[0, 1, 0]).sum().abs() < 1e-6


if __name__ == '__main__':
    test()
