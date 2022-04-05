from typing import List, Tuple, Optional

from allennlp.nn.util import masked_softmax
from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation


@Attention.register("span_masking")
class SpanMaskingAttention(Attention):
    def __init__(
        self,
        vector_dim: int,
        matrix_dim: int,
        activation: Activation = None,
        span_mask_weight: float = 1.,
        normalize: bool = True,
    ) -> None:
        super().__init__(normalize)
        assert normalize

        self._weight_matrix = Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or Activation.by_name("linear")()
        self._span_mask_weight = span_mask_weight
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    @overrides
    def forward(
        self,
        vector: torch.Tensor,
        matrix: torch.Tensor,
        matrix_mask: torch.BoolTensor,
        span_index: Optional[List[Tuple[int]]] = None,
    ) -> torch.Tensor:
        # Shape of `span_index`: (batch_size, 2)

        # (batch_size, 1, matrix_dim)
        intermediate = vector.mm(self._weight_matrix).unsqueeze(1)
        # (batch_size, matrix_length)
        att_weight = self._activation(intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._bias)

        if span_index and self._span_mask_weight not in {None, 1.0}:
            # (batch_size, matrix_length)
            bias_mask = torch.full(
                (att_weight.size(0), att_weight.size(1)),
                fill_value=self._span_mask_weight,
            )

            # FIXME: this will be very slow
            for example_idx, span in enumerate(span_index):
                if span != [0, 0] and span != [None, None]:
                    bias_mask[example_idx, span[0]: span[1]] = 1.0

            bias_mask = bias_mask.to(intermediate.device)
            bias_mask = bias_mask.log()
            att_weight = bias_mask + att_weight

        att_prob = masked_softmax(att_weight, matrix_mask)

        return att_prob
