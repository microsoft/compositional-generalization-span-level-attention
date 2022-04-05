from allennlp.nn.util import masked_softmax
from overrides import overrides
import torch
from torch.nn.parameter import Parameter

from allennlp.modules.attention.attention import Attention
from allennlp.nn import Activation, util


@Attention.register("oracle_alignment")
class OracleAlignmentAttention(Attention):
    """
    Computes attention between a vector and a matrix using a bilinear attention function.  This
    function has a matrix of weights `W` and a bias `b`, and the similarity between the vector
    `x` and the matrix `y` is computed as `x^T W y + b`.

    Registered as an `Attention` with name "bilinear".

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `x^T W y + b` calculation.  Default is
        linear, i.e. no activation.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(
        self,
        matrix_dim: int = None,
        vector_dim: int = None
    ) -> None:
        super().__init__(normalize=True)

    @overrides
    def forward(
        self,
        vector: torch.Tensor,
        matrix: torch.Tensor,
        matrix_mask: torch.BoolTensor,
        target_to_source_alignment: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Args:
            vector: (batch_size, [vector_num], vector_dim)
            matrix: (batch_size, matrix_len, matrix_dim)
            matrix_mask: (batch_size, matrix_len)
            target_to_source_alignment: (batch_size, matrix_len)
        """
        if vector.ndimension() == 3:
            # (batch_size, vector_num, matrix_dim)
            target_to_source_alignment = target_to_source_alignment.unsqueeze(1).expand(-1, vector.size(1), -1)

        if self._normalize:
            target_to_source_alignment = target_to_source_alignment / (
                target_to_source_alignment.sum(dim=-1) +
                util.tiny_value_of_dtype(torch.float)
            ).unsqueeze(-1)

        return target_to_source_alignment
