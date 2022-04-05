from typing import Dict, Tuple, List, Any, Union, cast, Optional

from dataflow.core.linearize import seq_to_sexp
from overrides import overrides
import numpy
import logging

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

from allennlp.modules.attention import BilinearAttention
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU, Perplexity, SequenceAccuracy

from models.beam_search import BeamSearchWithStatesLogging
from models.modules.coverage_attention import CoverageAdditiveAttention
from models.parser import Parser
from models.sequence_metric import SequenceMatchingMetric, SequenceCategorizedMatchMetric
from models.structured_metric import StructuredRepresentationMetric
from models.stacked_lstm_cell import StackedLSTMCell


logger = logging.getLogger(__name__)


@Model.register("seq2seq_with_copy")
class Seq2SeqModelWithCopy(Model, Parser):
    """
    This is an implementation of [CopyNet](https://arxiv.org/pdf/1603.06393).
    CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
    that can copy tokens from the source sentence into the target sentence instead of
    generating all target tokens only from the target vocabulary.

    It is very similar to a typical seq2seq model used in neural machine translation
    tasks, for example, except that in addition to providing a "generation" score at each timestep
    for the tokens in the target vocabulary, it also provides a "copy" score for each
    token that appears in the source sentence. In other words, you can think of CopyNet
    as a seq2seq model with a dynamic target vocabulary that changes based on the tokens
    in the source sentence, allowing it to predict tokens that are out-of-vocabulary (OOV)
    with respect to the actual target vocab.

    # Parameters

    vocab : `Vocabulary`, required
        Vocabulary containing source and target vocabularies.
    source_embedder : `TextFieldEmbedder`, required
        Embedder for source side sequences
    encoder : `Seq2SeqEncoder`, required
        The encoder of the "encoder/decoder" model
    attention : `Attention`, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    beam_size : `int`, required
        Beam width to use for beam search prediction.
    max_decoding_steps : `int`, required
        Maximum sequence length of target predictions.
    target_embedding_dim : `int`, optional (default = 30)
        The size of the embeddings for the target vocabulary.
    copy_token : `str`, optional (default = '@COPY@')
        The token used to indicate that a target token was copied from the source.
        If this token is not already in your target vocabulary, it will be added.
    source_namespace : `str`, optional (default = 'source_tokens')
        The namespace for the source vocabulary.
    target_namespace : `str`, optional (default = 'target_tokens')
        The namespace for the target vocabulary.
    tensor_based_metric : `Metric`, optional (default = BLEU)
        A metric to track on validation data that takes raw tensors when its called.
        This metric must accept two arguments when called: a batched tensor
        of predicted token indices, and a batched tensor of gold token indices.
    token_based_metric : `Metric`, optional (default = None)
        A metric to track on validation data that takes lists of lists of tokens
        as input. This metric must accept two arguments when called, both
        of type `List[List[str]]`. The first is a predicted sequence for each item
        in the batch and the second is a gold sequence for each item in the batch.
    initializer : `InitializerApplicator`, optional
        An initialization strategy for the model weights.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention: Attention,
        use_coverage_attention: bool = False,
        target_embedder: Optional[Embedding] = None,
        target_embedding_dim: Optional[int] = None,
        beam_size: int = 5,
        max_decoding_steps: int = 100,
        decoder_hidden_dim: int = 256,
        num_decoder_layers: int = 1,
        decoder_dropout: float = 0.,
        enable_copy: bool = True,
        copy_token: str = "@COPY@",
        source_namespace: str = "source_tokens",
        target_namespace: str = "target_tokens",
        tie_target_embedding: bool = False,
        attention_regularization: Optional[str] = None,
        label_smoothing: Optional[float] = None,
        beam_search_output_attention: Optional[bool] = False,
        use_sketch_metric: Optional[bool] = False,
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:
        super().__init__(vocab)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace

        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)
        self._pad_index = self.vocab.get_token_index(
            self.vocab._padding_token, self._target_namespace
        )

        self._copy_index = self.vocab.add_token_to_namespace(copy_token, self._target_namespace)

        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        self._enable_copy = enable_copy
        if not enable_copy:
            logger.warning('Copy feature is disabled!')

        # Encoding modules.
        self._source_embedder = source_embedder
        self._encoder = encoder

        assert (
            bool(target_embedder) != bool(target_embedding_dim),
            "You may only either `target_embedder` or `target_embedding_dim`"
        )

        self._target_embedder = target_embedder or Embedding(
            num_embeddings=self._target_vocab_size, embedding_dim=target_embedding_dim
        )
        target_embedding_dim = self._target_embedder.get_output_dim()

        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_input_dim = target_embedding_dim + decoder_hidden_dim
        self.decoder_output_dim = self.decoder_hidden_dim = decoder_hidden_dim

        self._num_decoder_layers = num_decoder_layers
        self._decoder_cell = StackedLSTMCell(
            self.decoder_input_dim, self.decoder_output_dim, num_decoder_layers, decoder_dropout)

        self._decoder_state_init_linear = Linear(self.encoder_output_dim, self.decoder_output_dim)

        self._attention = attention
        self._use_coverage_attention = use_coverage_attention
        if use_coverage_attention:
            assert attention is None
            self._attention = CoverageAdditiveAttention(
                vector_dim=self.decoder_hidden_dim,
                matrix_dim=self.encoder_output_dim
            )

        self._decoder_dropout = nn.Dropout(decoder_dropout)

        self._decoder_attention_vec_linear = Linear(
            self.decoder_hidden_dim + self.encoder_output_dim, self.decoder_output_dim, bias=False)

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        if tie_target_embedding:
            self._output_generation_layer_bias = nn.Parameter(torch.Tensor(self._target_vocab_size))
            self._output_generation_layer_bias.data.zero_()
            self._output_generation_layer = (
                lambda x: nn.functional.linear(
                    x, self._target_embedder.weight, self._output_generation_layer_bias)
            )
        else:
            self._output_generation_layer = Linear(self.decoder_output_dim, self._target_vocab_size)

        self._copy_attention = BilinearAttention(self.decoder_output_dim, self.encoder_output_dim, normalize=False)

        # Attention regularization method
        # assert attention_regularization is None or attention_regularization.startswith('mse')
        self._attention_regularization = None
        if attention_regularization:
            self._attention_reg_weight = (
                float(attention_regularization.rpartition(':')[-1])
                if ':' in attention_regularization
                else 1.0
            )
            self._attention_regularization = attention_regularization

        self._normalize_target_attention_distribution = (
            attention_regularization is not None and
            'src_normalize' in attention_regularization
        )

        self._decode_with_oracle_alignments = (
            attention_regularization is not None and
            'oracle_align' in attention_regularization
        )

        if attention_regularization and 'complementary' in attention_regularization and self._normalize_target_attention_distribution:
            raise ValueError(f'conflict config: {attention_regularization}')

        self._label_smoothing = label_smoothing
        if label_smoothing and enable_copy:
            raise ValueError('label_smoothing does not support copying yet.')

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_search = BeamSearchWithStatesLogging(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )
        self._beam_search_output_attention = beam_search_output_attention

        self._bleu = BLEU(exclude_indices={self._pad_index, self._start_index, self._end_index})
        self._token_based_metric = SequenceCategorizedMatchMetric()

        self._sketch_metric = None
        if use_sketch_metric:
            self._sketch_metric = StructuredRepresentationMetric()

        initializer(self)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        source_tokens: TextFieldTensors,
        source_token_first_appearing_indices: torch.LongTensor,
        source_to_target_token_idx_map: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        target_tokens: TextFieldTensors = None,
        target_token_first_appearing_indices: torch.LongTensor = None,
        target_to_source_alignment: torch.FloatTensor = None,
        call_from_vae: bool = False
    ) -> Dict[str, torch.Tensor]:
        state = self._encode(source_tokens)

        state.update({
            'source_token_first_appearing_indices': source_token_first_appearing_indices,
            'source_to_target_token_idx_map': source_to_target_token_idx_map
        })

        if self._decode_with_oracle_alignments:
            assert target_to_source_alignment is not None, \
                'The model is set to always use oracle alignments without providing the oracle alignments'
            target_to_source_alignment_mask = (target_to_source_alignment != -1)  # .to(torch.float)
            target_to_source_alignment_ = target_to_source_alignment * target_to_source_alignment_mask
            # omit alignments for the leading BOS token
            state['target_to_source_alignment'] = target_to_source_alignment_[:, 1:]

        if target_tokens:
            # state.update({
            #     'target_tokens': target_tokens,
            #     'target_token_first_appearing_indices': target_token_first_appearing_indices
            # })

            state = self._init_decoder_state(state)

            # shape: (batch_size, target_sequence_length)
            target_tokens_mask = util.get_text_field_mask(target_tokens)

            output_dict = self._forward_loss(
                target_tokens, target_tokens_mask,
                target_token_first_appearing_indices,
                target_to_source_alignment,
                state
            )
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

        if not self.training and not call_from_vae:
            state = self._init_decoder_state(state)
            predictions = self.inference(state, return_attention=self._beam_search_output_attention)
            output_dict.update(predictions)

            if target_tokens:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]

                batch_size, top_k, max_decoding_timestep = top_k_predictions.size()

                # shape: (batch_size, target_sequence_length)
                gold_token_ids = self._gather_extended_gold_tokens(
                    target_tokens["tokens"]["tokens"],
                    source_token_first_appearing_indices, target_token_first_appearing_indices
                )

                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]

                if self._bleu:
                    self._bleu(best_predictions, gold_token_ids)

                if self._token_based_metric:
                    hyp_tokens: List[List[str]] = self._get_predicted_tokens(best_predictions, metadata, n_best=1)
                    ref_tokens: List[List[str]] = self._get_predicted_tokens(gold_token_ids[:, 1:], metadata, n_best=1)
                    self._token_based_metric(hyp_tokens, ref_tokens, [x['tags'] for x in metadata])

                    if self._sketch_metric:
                        ref_sexp_list = [
                            seq_to_sexp(ref_sexp_tokens)
                            for ref_sexp_tokens
                            in ref_tokens
                        ]
                        hyp_sexp_list = []
                        for hyp_sexp_tokens in hyp_tokens:
                            try:
                                hyp_sexp = seq_to_sexp(hyp_sexp_tokens)
                            except:
                                hyp_sexp = []

                            hyp_sexp_list.append(hyp_sexp)

                        self._sketch_metric(
                            hyp_sexp_list,
                            None,
                            ref_sexp_list,
                            [x['tags'] for x in metadata]
                        )

        return output_dict

    @property
    def _use_pretrained_encoder(self):
        return isinstance(
            getattr(self._source_embedder, 'token_embedder_tokens'),
            PretrainedTransformerEmbedder
        )

    def _encode(self, source_tokens: TextFieldTensors) -> Dict[str, torch.Tensor]:
        """
        Encode source input sentences.
        """
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = self._encoder(embedded_input, source_mask)

        return {"source_tokens_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, encoder_output_dim)
        if self._use_pretrained_encoder:
            # use the leading [CLS] token embedding
            encoder_last_states = state['encoder_outputs'][:, 0]
        else:
            encoder_last_states = util.get_final_encoder_states(
                state['encoder_outputs'], state['source_tokens_mask'], self._encoder.is_bidirectional())

        # shape: (batch_size, 1, encoder_output_dim)
        decoder_init_state = torch.tanh(self._decoder_state_init_linear(encoder_last_states)).unsqueeze(1)
        batch_size = state['encoder_outputs'].size(0)

        if self._num_decoder_layers > 1:
            num_reminder_layers = self._num_decoder_layers - 1
            zeros = torch.zeros(
                self.decoder_hidden_dim, device=self.device
            )[None, None, :].expand(batch_size, num_reminder_layers, -1).contiguous()

            decoder_init_state = torch.cat([decoder_init_state, zeros], dim=1)

        decoder_init_cell = torch.zeros(
            self.decoder_output_dim,
            device=self.device
        )[None, None, :].expand(batch_size, self._num_decoder_layers, -1).contiguous()

        state.update({
            'decoder_hidden': decoder_init_state,
            'decoder_cell': decoder_init_cell,
            'decoder_attentional_vec': torch.zeros(batch_size, self.decoder_hidden_dim, device=self.device)
        })

        if self._use_coverage_attention:
            source_sequence_length = state['encoder_outputs'].size(1)
            state['coverage_context'] = torch.zeros(batch_size, source_sequence_length, device=self.device)

        return state

    def _forward_loss(
        self,
        target_tokens: TextFieldTensors,
        target_tokens_mask: torch.BoolTensor,
        target_token_first_appearing_indices: torch.LongTensor,
        target_to_source_alignment: Optional[torch.LongTensor],
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size = target_tokens_mask.size(0)

        decoder_forward_output = self._forward_decode(
            target_tokens, target_tokens_mask, target_token_first_appearing_indices,
            state
        )

        target_tokens_id = target_tokens['tokens']['tokens']

        # shape: (batch_size, target_sequence_length - 1)
        step_log_likelihoods = self._get_forward_log_likelihoods(
            decoder_forward_output['generation_scores'],
            decoder_forward_output['copy_scores'],
            state['source_tokens_mask'],
            target_tokens_id[:, 1:], target_tokens_mask[:, 1:],
            decoder_forward_output['target_token_source_position_mask'][:, 1:]
        )

        # Sum of step log-likelihoods.
        # shape: (batch_size,)

        tgt_log_likelihood = step_log_likelihoods.sum(dim=-1)
        # The loss is the negative log-likelihood, averaged over the batch.
        loss = -tgt_log_likelihood.sum() / batch_size

        # output_dict = {'decoder_loss': loss.item()}
        output_dict = {
            'log_prob': tgt_log_likelihood
        }

        # apply attention regularization
        if (
            self._attention_regularization and
            self.training and
            'mse' in self._attention_regularization
        ):
            # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
            attention_weights = decoder_forward_output['attention_weights']
            # shape: (batch_size, target_sequence_length, source_sequence_length)
            target_to_source_alignment_mask = (target_to_source_alignment != -1) #.to(torch.float)
            target_to_source_alignment = target_to_source_alignment * target_to_source_alignment_mask

            # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
            target_attention_distribution = target_to_source_alignment[:, 1:]
            target_attention_distribution_mask = target_to_source_alignment_mask[:, 1:]
            # shape: (batch_size, target_sequence_length - 1)
            target_attention_distribution_target_timestep_mask = target_attention_distribution_mask.any(dim=-1)

            if self._normalize_target_attention_distribution:
                target_attention_distribution = target_attention_distribution / (
                    target_attention_distribution.sum(dim=-1) +
                    util.tiny_value_of_dtype(torch.float) * ~target_attention_distribution_target_timestep_mask
                ).unsqueeze(-1)

            # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
            att_reg = nn.MSELoss(reduction='none')(
                attention_weights,
                target_attention_distribution
            ) * target_attention_distribution_mask

            # shape: (batch_size, target_sequence_length - 1)
            num_attn_regularized_src_tokens = target_attention_distribution_mask.sum(dim=-1)
            # shape: (batch_size, )
            # (target_to_source_alignment[:, 1:].sum(dim=-1) > 0).sum(dim=-1)
            num_attn_regularized_tgt_tokens = target_attention_distribution_target_timestep_mask.sum(dim=-1)

            # shape: (batch_size)
            att_reg = (
                att_reg.sum(dim=-1)
                # / (
                #     num_attn_regularized_src_tokens +
                #     util.tiny_value_of_dtype(torch.float)
                # )
            ).sum(dim=-1) / (
                num_attn_regularized_tgt_tokens +
                util.tiny_value_of_dtype(torch.float)
            )

            # att_reg = (
            #     att_reg.sum(dim=-1) /  # over `source_sequence_length`
            #     (
            #         num_attn_regularized_src_tokens +
            #         util.tiny_value_of_dtype(torch.float)
            #     )
            # ).sum(dim=-1) / (
            #     num_attn_regularized_tgt_tokens +
            #     util.tiny_value_of_dtype(torch.float)
            # )

            # shape: scalar
            att_reg_loss = att_reg.mean() * self._attention_reg_weight

            loss = loss + att_reg_loss

            # output_dict.update({
            #     'att_reg_loss_val': att_reg_loss.item()
            # })

        output_dict["loss"] = loss

        return output_dict

    def _forward_decode(
        self,
        target_tokens: TextFieldTensors,
        target_tokens_mask: torch.BoolTensor,
        target_token_first_appearing_indices: torch.LongTensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, target_sequence_length = target_tokens["tokens"]["tokens"].size()

        # shape: (batch_size, source_sequence_length)
        source_token_first_appearing_indices = state['source_token_first_appearing_indices']

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        # We use this to fill in the copy index when the previous input was copied.
        # shape: (batch_size,)
        copy_symbols = cast(torch.LongTensor, torch.full(
            (batch_size, 1), fill_value=self._copy_index, dtype=torch.long, device=self.device
        ))

        # shape: (batch_size, target_sequence_length, source_sequence_length)
        target_token_source_position_mask = (
            source_token_first_appearing_indices.unsqueeze(1) ==
            target_token_first_appearing_indices.unsqueeze(-1)
        ) & target_tokens_mask.unsqueeze(-1)

        # shape: (batch_size, target_sequence_length)
        target_token_copiable_mask = (target_token_source_position_mask.sum(dim=-1) > 0)

        # shape: (batch_size, target_sequence_length)
        target_tokens_id = target_tokens["tokens"]["tokens"]

        # shape: (batch_size, target_sequence_length)
        is_target_token_copied_only = (
            (target_tokens_id == self._oov_index) & target_token_copiable_mask
        ).long()

        # shape: (batch_size, target_sequence_length)
        if self._enable_copy:
            target_token_symbols = is_target_token_copied_only * copy_symbols + (
                1 - is_target_token_copied_only) * target_tokens_id
        else:
            target_token_symbols = target_tokens_id

        # Initialize attention vector
        att_tm1 = torch.zeros(batch_size, self.decoder_hidden_dim, device=self.device)

        att_vecs = []
        att_weights = []
        copy_scores = []

        for time_step in range(num_decoding_steps):
            # shape: (batch_size, )
            y_tm1 = target_token_symbols[:, time_step]

            # shape: (batch_size, target_embedding_dim)
            y_tm1_embed = self._target_embedder(y_tm1)

            # shape: (batch_size, target_embedding_dim + decoder_hidden_dim)
            x_t = torch.cat([y_tm1_embed, att_tm1], dim=-1)

            # Update the decoder state by taking a step through the RNN.
            # `decoder_output` shape: (batch_size, decoder_hidden_size)
            # `att_weight_t` shape: (batch_size, source_input_length)
            att_t, state, att_weight_t = self._decoder_step(x_t, state, return_attention=True, time_step=time_step)

            step_copy_scores = self._get_copy_scores(att_t, state)

            att_vecs.append(att_t)
            att_weights.append(att_weight_t)
            copy_scores.append(step_copy_scores)

            att_tm1 = att_t

        # shape: (batch_size, target_sequence_length - 1, decoder_hidden_size)
        decoder_att_vec = torch.stack(att_vecs, dim=1)

        # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
        att_weights = torch.stack(att_weights, dim=1)

        # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
        copy_scores = torch.stack(copy_scores, dim=1)

        # shape: (batch_size, target_sequence_length - 1, decoder_hidden_size)
        generation_scores = self._get_generation_scores(decoder_att_vec)

        return {
           'attention_weights': att_weights,
           'copy_scores': copy_scores,
           'generation_scores': generation_scores,
           'target_token_source_position_mask':target_token_source_position_mask
        }

    def _decoder_step(
        self,
        x_t: torch.Tensor,
        state_tm1: Dict[str, torch.Tensor],
        time_step: int,
        return_attention: bool = False
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
        decoder_output_t, (h_t, cell_t) = self._decoder_cell(
            x_t, (state_tm1["decoder_hidden"], state_tm1["decoder_cell"])
        )

        attention_extra_args = {}
        if self._use_coverage_attention:
            attention_extra_args = {
                'coverage_vector': state_tm1['coverage_context']
            }
        elif self._decode_with_oracle_alignments:
            max_oracle_target_sequence_len = state_tm1['target_to_source_alignment'].size(1)
            if not self.training and time_step >= max_oracle_target_sequence_len:
                time_step = max_oracle_target_sequence_len - 1

            attention_extra_args = {
                # (batch_size, max_input_sequence_length)
                'target_to_source_alignment': state_tm1['target_to_source_alignment'][:, time_step]
            }

        # shape: (batch_size, max_input_sequence_length)
        attention_weights = self._attention(
            decoder_output_t, state_tm1["encoder_outputs"], state_tm1["source_tokens_mask"],
            **attention_extra_args
        )

        # shape: (batch_size, encoder_output_dim)
        context_vec = util.weighted_sum(state_tm1["encoder_outputs"], attention_weights)

        # shape: (batch_size, decoder_output_dim)
        attentional_vec = self._decoder_dropout(torch.tanh(
            self._decoder_attention_vec_linear(torch.cat([decoder_output_t, context_vec], dim=-1))
        ))

        state_tm1.update({
            'decoder_hidden': h_t,
            'decoder_cell': cell_t,
            'decoder_attentional_vec': attentional_vec
        })

        if self._use_coverage_attention:
            state_tm1['coverage_context'] = state_tm1['coverage_context'] + attention_weights

        state_t = state_tm1

        return_tuple = (attentional_vec, state_t)
        if return_attention:
            return_tuple = return_tuple + (attention_weights, )

        return return_tuple

    def _get_generation_scores(self, att_vec: torch.Tensor) -> torch.Tensor:
        return self._output_generation_layer(att_vec)

    def _get_copy_scores(self, att_vec: torch.Tensor, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        copy_scores = self._copy_attention(att_vec, state['encoder_outputs'])

        return copy_scores

    def _get_forward_log_likelihoods(
        self,
        generation_scores: torch.Tensor,
        copy_scores: torch.Tensor,
        source_token_mask: torch.BoolTensor,
        target_tokens_id: torch.LongTensor,
        target_tokens_mask: torch.BoolTensor,
        target_token_source_position_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log-likelihood contribution from a single timestep.

        # Parameters

        generation_scores : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length - 1, target_vocab_size)`
        generation_scores_mask : `torch.BoolTensor`
            Shape: `(batch_size, target_vocab_size)`. This is just a tensor of 1's.
        copy_scores : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length - 1, source_sequence_length)`
        target_token_ids : `torch.LongTensor`
            Shape: `(batch_size, target_sequence_length - 1)`
        target_token_source_position_mask : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length - 1, source_sequence_length)`

        # Returns

        Tuple[torch.Tensor, torch.Tensor]
            Shape: `(batch_size,), (batch_size, trimmed_source_length)`
        """
        batch_size, num_decoding_time_step, target_vocab_size = generation_scores.size()

        if self._enable_copy:
            generation_scores_mask = torch.full(
                (batch_size, 1, target_vocab_size),
                fill_value=True, dtype=torch.bool, device=self.device
            )

            # shape: (batch_size, target_sequence_length - 1, target_vocab_size + source_sequence_length)
            scores_mask = torch.cat(
                [generation_scores_mask, source_token_mask.unsqueeze(1)], dim=-1
            ).expand(-1, num_decoding_time_step, -1)

            # shape: (batch_size, target_sequence_length - 1, target_vocab_size + source_sequence_length)
            all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        else:
            all_scores = generation_scores
            scores_mask = None

        if self._label_smoothing:
            assert not self._enable_copy

            # (batch_size, )
            step_log_likelihoods = -util.sequence_cross_entropy_with_logits(
                logits=generation_scores,
                targets=target_tokens_id.contiguous(),
                weights=target_tokens_mask,
                average=None,
                label_smoothing=self._label_smoothing
            ) * target_tokens_mask.sum(dim=-1)
        else:
            # Globally normalize generation and copy scores.
            # shape: (batch_size, target_vocab_size + source_sequence_length)
            log_probs = util.masked_log_softmax(all_scores, mask=scores_mask, dim=-1)

            if self._enable_copy:
                # Calculate the log probability for copying each token in the source sentence
                # that matches the current target token. We use the sum of these copy probabilities
                # for matching tokens in the source sentence to get the total probability
                # for the target token.
                # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
                tgt_copy_log_probs = (
                    log_probs[:, :, target_vocab_size:]
                    + (
                        target_token_source_position_mask.to(log_probs.dtype) + util.tiny_value_of_dtype(log_probs.dtype)
                    ).log()
                )

                # This mask ensures that item in the batch has a non-zero generation probabilities
                # for this timestep only when the gold target token is not OOV or there are no
                # matching tokens in the source sentence.
                # shape: (batch_size, target_sequence_length - 1)
                generation_mask = (target_tokens_id != self._oov_index) | (target_token_source_position_mask.sum(-1) == 0)

                log_generation_mask = (generation_mask + util.tiny_value_of_dtype(log_probs.dtype)).log().unsqueeze(-1)

                # Now we get the generation score for the gold target token.
                # shape: (batch_size, target_sequence_length - 1, 1)
                tgt_generation_log_probs = log_probs.gather(
                    dim=-1,
                    index=target_tokens_id.unsqueeze(-1)
                ) + log_generation_mask

                # ... and add the copy score to get the step log likelihood.
                # shape: (batch_size, target_sequence_length - 1, 1 + source_sequence_length)
                combined_gen_and_copy = torch.cat((tgt_generation_log_probs, tgt_copy_log_probs), dim=-1)

                # shape: (batch_size, target_sequence_length - 1)
                step_log_likelihoods = util.logsumexp(combined_gen_and_copy, dim=-1) * target_tokens_mask
            else:
                # shape: (batch_size, target_sequence_length - 1)
                step_log_likelihoods = log_probs.gather(
                    dim=-1,
                    index=target_tokens_id.unsqueeze(-1)
                ).unsqueeze(-1) * target_tokens_mask

        return step_log_likelihoods

    def inference(
        self,
        state: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Perform beam search inference
        """

        batch_size, source_sequence_length = state['source_tokens_mask'].size()

        # shape: (batch_size,)
        bos = torch.full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long, device=self.device
        )

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities, *states = self._beam_search.search(
            bos, state, self._beam_search_step, return_state=return_attention
        )

        output_dict = {
            "predicted_log_probs": log_probabilities,
            "predictions": all_top_k_predictions,
        }

        # extract attention maps
        if return_attention:
            states = states[0]
            # shape (batch_size, target_sequence_len, source_sequence_len)
            best_prediction_attention_weights = torch.stack(
                [state['attention_weights'][:, 0] for state in states],
                dim=1
            )

            output_dict['best_prediction_attention_weights'] = best_prediction_attention_weights

        # print(output_dict)

        return output_dict

    def _beam_search_step(
        self,
        prev_predictions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        timestep: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Take step during beam search.

        This function is what gets passed to the `BeamSearch.search` method. It takes
        predictions from the last timestep and the current state and outputs
        the log probabilities assigned to tokens for the next timestep, as well as the updated
        state.

        Since we are predicting tokens out of the extended vocab (target vocab + all unique
        tokens from the source sentence), this is a little more complicated that just
        making a forward pass through the model. The output log probs will have
        shape `(group_size, target_vocab_size + trimmed_source_length)` so that each
        token in the target vocab and source sentence are assigned a probability.

        Note that copy scores are assigned to each source token based on their position, not unique value.
        So if a token appears more than once in the source sentence, it will have more than one score.
        Further, if a source token is also part of the target vocab, its final score
        will be the sum of the generation and copy scores. Therefore, in order to
        get the score for all tokens in the extended vocab at this step,
        we have to combine copy scores for re-occuring source tokens and potentially
        add them to the generation scores for the matching token in the target vocab, if
        there is one.

        So we can break down the final log probs output as the concatenation of two
        matrices, A: `(group_size, target_vocab_size)`, and B: `(group_size, trimmed_source_length)`.
        Matrix A contains the sum of the generation score and copy scores (possibly 0)
        for each target token. Matrix B contains left-over copy scores for source tokens
        that do NOT appear in the target vocab, with zeros everywhere else. But since
        a source token may appear more than once in the source sentence, we also have to
        sum the scores for each appearance of each unique source token. So matrix B
        actually only has non-zero values at the first occurence of each source token
        that is not in the target vocab.

        # Parameters

        last_predictions : `torch.Tensor`
            Shape: `(group_size,)`

        state : `Dict[str, torch.Tensor]`
            Contains all state tensors necessary to produce generation and copy scores
            for next step.

        Notes
        -----
        `group_size` != `batch_size`. In fact, `group_size` = `batch_size * beam_size`.
        """
        group_size, source_sequence_length = state["source_token_first_appearing_indices"].size()

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied,
        # there will be all zeros.

        # shape: (group_size,)
        prev_copied_only_mask = prev_predictions >= self._target_vocab_size

        # If the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        copy_input_choices = torch.full(
            (group_size,), fill_value=self._copy_index, dtype=torch.long, device=self.device
        )
        # shape: (group_size,)
        y_tm1 = prev_predictions * ~prev_copied_only_mask + copy_input_choices * prev_copied_only_mask

        # shape: (group_size, tgt_embed_size)
        y_tm1_embed = self._target_embedder(y_tm1)

        x_t = torch.cat([y_tm1_embed, state['decoder_attentional_vec']], dim=-1)

        # Update the decoder state by taking a step through the RNN.
        att_vec, state, attention_weights = self._decoder_step(
            x_t, state,
            return_attention=True,
            time_step=timestep
        )
        state['attention_weights'] = attention_weights

        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(att_vec)
        # shape: (group_size, target_vocab_size)
        generation_mask = torch.full(generation_scores.size(), True, dtype=torch.bool, device=self.device)
        all_scores = generation_scores
        scores_mask = generation_mask

        if self._enable_copy:
            # Get the un-normalized copy scores for each token in the source sentence,
            # excluding the start and end tokens.
            # shape: (group_size, source_sequence_length)
            copy_scores = self._get_copy_scores(att_vec, state)

            # Concat un-normalized generation and copy scores.
            # shape: (batch_size, target_vocab_size + source_sequence_length)
            all_scores = torch.cat((generation_scores, copy_scores), dim=-1)

            # shape: (group_size, source_sequence_length)
            copy_mask = state["source_tokens_mask"]

            # shape: (group_size, target_vocab_size + source_sequence_length)
            scores_mask = torch.cat(
                (
                    # shape: (group_size, target_vocab_size)
                    generation_mask,
                    # shape: (group_size, source_sequence_length)
                    copy_mask,
                ),
                dim=-1,
            )

        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size [+ trimmed_source_length])
        log_probs = util.masked_log_softmax(all_scores, mask=scores_mask)

        if self._enable_copy:
            # shape: (group_size, target_vocab_size), (group_size, trimmed_source_length)
            generation_log_probs, copy_log_probs = log_probs.split(
                [self._target_vocab_size, source_sequence_length], dim=-1
            )

            # We now have normalized generation and copy scores, but to produce the final
            # score for each token in the extended vocab, we have to go through and add
            # the copy scores to the generation scores of matching target tokens, and sum
            # the copy scores of duplicate source tokens.
            # shape: (group_size, target_vocab_size + trimmed_source_length)
            final_log_probs = self._beam_search_step_aggregate_log_probs(generation_log_probs, copy_log_probs, state)
        else:
            final_log_probs = log_probs

        return final_log_probs, state

    def _beam_search_step_aggregate_log_probs(
        self,
        generation_log_probs: torch.Tensor,
        copy_log_probs: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine copy probabilities with generation probabilities for matching tokens.

        # Parameters

        generation_log_probs : `torch.Tensor`
            Shape: `(group_size, target_vocab_size)`
        copy_log_probs : `torch.Tensor`
            Shape: `(group_size, source_sequence_length)`
        state : `Dict[str, torch.Tensor]`

        # Returns

        torch.Tensor
            Shape: `(group_size, target_vocab_size + source_sequence_length)`.
        """
        source_token_first_appearing_indices = state["source_token_first_appearing_indices"]
        group_size, source_sequence_length = source_token_first_appearing_indices.size()

        # shape: [(group_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(source_sequence_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]

            # `source_to_target` is a matrix of shape (group_size, source_sequence_length)
            # where element (i, j) is the vocab index of the target token that matches the jth
            # source token in the ith group, if there is one, or the index of the OOV symbol otherwise.
            # We'll use this to add copy scores to corresponding generation scores.
            # shape: (group_size,)
            source_to_target_idx_map_slice = state["source_to_target_token_idx_map"][:, i]

            # The OOV index in the source_to_target_slice indicates that the source
            # token is not in the target vocab, so we don't want to add that copy score
            # to the OOV token.
            # shape: (group_size,)
            src_token_in_tgt_vocab_mask = (source_to_target_idx_map_slice != self._oov_index)
            copy_log_probs_to_add_to_gen_probs = (
                copy_log_probs_slice
                + (
                    src_token_in_tgt_vocab_mask.to(copy_log_probs_slice.dtype)
                    + util.tiny_value_of_dtype(copy_log_probs_slice.dtype)
                ).log()
            )

            # shape: (group_size, 1)
            copy_log_probs_to_add_to_gen_probs = copy_log_probs_to_add_to_gen_probs.unsqueeze(-1)

            # shape: (group_size, 1)
            selected_generation_log_probs = generation_log_probs.gather(
                dim=-1, index=source_to_target_idx_map_slice.unsqueeze(-1)
            )

            # shape: (group_size,)
            combined_scores = util.logsumexp(
                torch.cat((selected_generation_log_probs, copy_log_probs_to_add_to_gen_probs), dim=1)
            )

            generation_log_probs = generation_log_probs.scatter(
                dim=-1, index=source_to_target_idx_map_slice.unsqueeze(-1), src=combined_scores.unsqueeze(-1)
            )

            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurrence of this particular source token, we add the log_probs from all other
            # occurrences, otherwise we zero it out since it was already accounted for.
            if i < (source_sequence_length - 1):
                # Sum copy scores from future occurrences of source token.
                # shape: (group_size, trimmed_source_length - i)
                source_future_occurrences_mask = (
                    source_token_first_appearing_indices[:, (i + 1):] ==
                    source_token_first_appearing_indices[:, i].unsqueeze(-1)
                ) & state['source_tokens_mask'][:, (i+1):]

                # shape: (group_size, trimmed_source_length - i)
                future_copy_log_probs = (
                    copy_log_probs[:, (i + 1):] +
                    (
                        source_future_occurrences_mask + util.tiny_value_of_dtype(copy_log_probs.dtype)
                    ).log()
                )

                # shape: (group_size, 1 + trimmed_source_length - i)
                combined = torch.cat(
                    [copy_log_probs_slice.unsqueeze(-1), future_copy_log_probs], dim=-1
                )

                # shape: (group_size,)
                copy_log_probs_slice = util.logsumexp(combined)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (group_size, i)
                source_previous_occurrences = (
                    source_token_first_appearing_indices[:, 0:i] ==
                    source_token_first_appearing_indices[:, i].unsqueeze(-1)
                )
                # shape: (group_size,)
                duplicate_mask = source_previous_occurrences.sum(dim=-1) == 0
                copy_log_probs_slice = (
                    copy_log_probs_slice
                    + (duplicate_mask + util.tiny_value_of_dtype(copy_log_probs_slice.dtype)).log()
                )

            # Finally, we zero-out copy scores that we added to the generation scores
            # above so that we don't double-count them.
            # shape: (group_size,)
            left_over_copy_log_probs = (
                copy_log_probs_slice
                + (
                    ~src_token_in_tgt_vocab_mask
                    + util.tiny_value_of_dtype(copy_log_probs_slice.dtype)
                ).log()
            )

            modified_log_probs_list.append(left_over_copy_log_probs.unsqueeze(-1))

        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def _get_predicted_tokens(
        self,
        predicted_indices: Union[torch.Tensor, numpy.ndarray],
        batch_metadata: List[Any],
        n_best: int = None,
    ) -> List[Union[List[List[str]], List[str]]]:
        """
        Convert predicted indices into tokens.

        predicted_indices: (batch_size, top-K, max_time_step)

        If `n_best = 1`, the result type will be `List[List[str]]`. Otherwise the result
        type will be `List[List[List[str]]]`.
        """
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()

        if predicted_indices.ndim == 2:
            predicted_indices = predicted_indices[:, None, :]

        predicted_tokens: List[Union[List[List[str]], List[str]]] = []
        for top_k_predictions, metadata in zip(predicted_indices, batch_metadata):
            batch_predicted_tokens: List[List[str]] = []
            for indices in top_k_predictions[:n_best]:
                tokens: List[str] = []
                indices = list(indices)
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                for index in indices:
                    if index >= self._target_vocab_size:
                        adjusted_index = index - self._target_vocab_size
                        token = metadata["source_tokens"][adjusted_index]
                    else:
                        token = self.vocab.get_token_from_index(index, self._target_namespace)
                    tokens.append(token)
                batch_predicted_tokens.append(tokens)
            if n_best == 1:
                predicted_tokens.append(batch_predicted_tokens[0])
            else:
                predicted_tokens.append(batch_predicted_tokens)
        return predicted_tokens

    def _gather_extended_gold_tokens(
        self,
        target_token_ids: torch.LongTensor,
        source_token_first_appearing_indices: torch.Tensor,
        target_token_fist_appearing_indices: torch.Tensor,
    ) -> torch.LongTensor:
        """
        Modify the gold target tokens relative to the extended vocabulary.

        For gold targets that are OOV but were copied from the source, the OOV index
        will be changed to the index of the first occurence in the source sentence,
        offset by the size of the target vocabulary.

        # Parameters

        target_tokens : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length)`.
        source_token_ids : `torch.Tensor`
            Shape: `(batch_size, trimmed_source_length)`.
        target_token_ids : `torch.Tensor`
            Shape: `(batch_size, target_sequence_length)`.

        # Returns

        torch.Tensor
            Modified `target_tokens` with OOV indices replaced by offset index
            of first match in source sentence.
        """
        batch_size, target_sequence_length = target_token_ids.size()
        trimmed_source_length = source_token_first_appearing_indices.size(1)
        # Only change indices for tokens that were OOV in target vocab but copied from source.
        # shape: (batch_size, target_sequence_length)
        oov = target_token_ids == self._oov_index
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_source_token_ids = source_token_first_appearing_indices.unsqueeze(1).expand(
            batch_size, target_sequence_length, trimmed_source_length
        )
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        expanded_target_token_ids = target_token_fist_appearing_indices.unsqueeze(-1).expand(
            batch_size, target_sequence_length, trimmed_source_length
        )
        # shape: (batch_size, target_sequence_length, trimmed_source_length)
        matches = expanded_source_token_ids == expanded_target_token_ids
        # shape: (batch_size, target_sequence_length)
        copied = matches.sum(-1) > 0
        # shape: (batch_size, target_sequence_length)
        mask = oov & copied
        # shape: (batch_size, target_sequence_length)
        first_match = ((matches.cumsum(-1) == 1) & matches).to(torch.uint8).argmax(-1)
        # shape: (batch_size, target_sequence_length)
        new_target_token_ids = (
            target_token_ids * ~mask + (first_match.long() + self._target_vocab_size) * mask
        )

        return new_target_token_ids

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}

        if not self.training:
            if self._bleu:
                all_metrics.update(self._bleu.get_metric(reset=reset))
            if self._token_based_metric:
                all_metrics.update(self._token_based_metric.get_metric(reset=reset))
            if self._sketch_metric:
                all_metrics.update(self._sketch_metric.get_metric(reset=reset))

        return all_metrics

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        batch_pred_tokens = self._get_predicted_tokens(
            output_dict["predictions"], output_dict["metadata"]
        )

        del output_dict["predictions"]
        # noinspection PyTypeChecker
        output_dict["predictions"] = [
            [
                {
                    'tokens': pred_tokens
                }
                for pred_tokens
                in example_pred_tokens
            ]
            for example_pred_tokens
            in batch_pred_tokens
        ]

        return output_dict
