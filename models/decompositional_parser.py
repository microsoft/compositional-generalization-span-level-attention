import itertools
from functools import partial
from typing import Dict, Tuple, List, Any, Union, cast, Optional
import copy
import collections

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataloader import TensorDict
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.predictors import Predictor
from dataflow.core.linearize import seq_to_sexp
from overrides import overrides
import numpy
import scipy
import logging

import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.modules.attention import BilinearAttention, DotProductAttention
from allennlp.common.util import START_SYMBOL, END_SYMBOL, JsonDict
from allennlp.data import TextFieldTensors, Vocabulary, Batch, Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, FeedForward
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util, Activation, Initializer
from allennlp.training.metrics import Metric, BLEU, Perplexity, SequenceAccuracy
from allennlp.nn.beam_search import BeamSearch, StateType

from dataflow.core.sexp import Sexp
# from dataflow.core.lispress import parse_lispress

from models import utils
from models.beam_search import BeamSearchWithStatesLogging
from models.decompositional_parser_reader import DecompositionalParserReader, CHILD_DERIVATION_STEP_MARKER, \
    index_child_derivation_step_marker
from models.span_masking_attention import SpanMaskingAttention
from models.oracle_metric import OracleMetric
from models.sequence_metric import SequenceMatchingMetric, SequenceCategorizedMatchMetric
from models.span_masking_attention import SpanMaskingAttention
from models.structured_metric import StructuredRepresentationMetric
from models.stacked_lstm_cell import StackedLSTMCell


LogicalFormTensors = TextFieldTensors


logger = logging.getLogger(__name__)


DerivationTensorDictList = List[Dict[str, Union[torch.Tensor, TextFieldTensors]]]


@Model.register("decompositional")
class DecompositionalParser(Model):
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
        dataset_reader: DatasetReader,
        source_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        attention: Attention,
        beam_size: int,
        max_decoding_steps: int,
        target_embedding_dim: int = 128,
        decoder_hidden_dim: int = 256,
        num_decoder_layers: int = 1,
        decoder_dropout: float = 0.,
        copy_token: str = "@COPY@",
        source_namespace: str = "source_tokens",
        target_namespace: str = "target_tokens",
        attention_regularization: Optional[str] = None,
        sketch_level_attention_regularization_only: Optional[bool] = None,
        parse_sketch_only: bool = False,
        beam_search_method: str = 'separate',
        dump_full_beam: bool = False,
        beam_search_output_attention: Optional[bool] = False,
        root_derivation_compute_span_scores: Optional[bool] = True,
        child_derivation_compute_span_scores: Optional[bool] = True,
        sketch_parser_no_source_span_prediction: Optional[bool] = False,  # deparated!
        child_derivation_use_separate_decoder: Optional[bool] = False,
        decoders_share_output_layers: Optional[bool] = True,
        decoders_share_target_embedding: Optional[bool] = True,
        child_derivation_use_separate_encoder: Optional[bool] = False,
        child_derivation_use_root_utterance_encoding: Optional[bool] = False,
        use_parent_state_gating: Optional[bool] = False,
        stop_gradient_from_parent_state: Optional[bool] = False,
        force_decoding_sketch: Optional[bool] = False,
        debug: Optional[bool] = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
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
        self._child_derivation_step_marker_idx = self.vocab.get_token_index(CHILD_DERIVATION_STEP_MARKER, self._target_namespace)

        self._copy_index = self.vocab.add_token_to_namespace(copy_token, self._target_namespace)

        self._target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # Encoding modules.
        self._source_embedder = source_embedder
        self._encoder = encoder

        self._child_derivation_use_separate_encoder = child_derivation_use_separate_encoder
        if child_derivation_use_separate_encoder:
            logger.info('Child derivation predictor uses separate encoder')
            self._child_encoder = copy.deepcopy(encoder)
            self._child_source_embedder = copy.deepcopy(source_embedder)

        self._child_derivation_use_root_utterance_encoding = child_derivation_use_root_utterance_encoding
        if child_derivation_use_root_utterance_encoding:
            if child_derivation_use_separate_encoder:
                raise ConfigurationError(
                    'Incompatible config values: child_derivation_use_separate_encoder and '
                    'child_derivation_use_root_utterance_encoding'
                )

            if child_derivation_compute_span_scores:
                raise ConfigurationError(
                    'child_derivation_compute_span_scores should be `false` if '
                    'child_derivation_use_root_utterance_encoding is `true`'
                )

        self._sketch_level_attention_regularization_only = sketch_level_attention_regularization_only
        if sketch_level_attention_regularization_only is not None:
            assert (
                child_derivation_use_root_utterance_encoding,
                "sketch_level_attention_regularization_only could only be activated "
                "when child_derivation_use_root_utterance_encoding is `true`"
            )

        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_input_dim = target_embedding_dim + decoder_hidden_dim
        self.decoder_output_dim = self.decoder_hidden_dim = decoder_hidden_dim

        # Target Token Embedding
        # This layer is shared between different derivation levels
        self._target_embedder = Embedding(
            num_embeddings=self._target_vocab_size, embedding_dim=target_embedding_dim
        )

        # Decoder Layers
        self._num_decoder_layers = num_decoder_layers
        self._decoder_cell = StackedLSTMCell(
            self.decoder_input_dim, self.decoder_output_dim, num_decoder_layers, decoder_dropout)

        self._decoder_state_init_linear = Linear(
            self.encoder_output_dim, self.decoder_output_dim)
        self._decoder_cell_init_linear = Linear(
            self.encoder_output_dim, self.decoder_output_dim)

        self._child_level_decoder_start_tag = nn.Parameter(torch.zeros(32))

        decoder_init_linear_input_dim = self.decoder_output_dim + self._child_level_decoder_start_tag.size(0)
        if not self._child_derivation_use_root_utterance_encoding and not parse_sketch_only:
            decoder_init_linear_input_dim += self.encoder_output_dim

        self._decoder_state_with_parent_init_linear_layer = Linear(
            decoder_init_linear_input_dim,
            self.decoder_output_dim
        )
        self._decoder_cell_with_parent_init_linear_layer = Linear(
            decoder_init_linear_input_dim,
            self.decoder_output_dim
        )

        self._decoder_state_with_parent_init_linear_layer.weight.data.zero_()
        self._decoder_state_with_parent_init_linear_layer.weight.data[:self.decoder_output_dim].fill_diagonal_(1.0)
        self._decoder_state_with_parent_init_linear_layer.bias.data.zero_()

        self._decoder_cell_with_parent_init_linear_layer.weight.data.zero_()
        self._decoder_cell_with_parent_init_linear_layer.weight.data[:self.decoder_output_dim].fill_diagonal_(1.0)
        self._decoder_cell_with_parent_init_linear_layer.bias.data.zero_()

        self._attention = attention

        self._decoder_dropout = nn.Dropout(decoder_dropout)

        self._decoder_attention_vec_linear = Linear(
            self.decoder_hidden_dim + self.encoder_output_dim, self.decoder_output_dim, bias=False)

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(self.decoder_output_dim, self._target_vocab_size)
        self._copy_attention = BilinearAttention(self.decoder_output_dim, self.encoder_output_dim, normalize=False)

        self._child_derivation_use_separate_decoder = child_derivation_use_separate_decoder
        if not decoders_share_output_layers:
            if not child_derivation_use_separate_decoder:
                raise ConfigurationError(
                    'Incompatible config: decoders_share_output_layers and '
                    'child_derivation_use_separate_decoder'
                )

        self._decoders_share_output_layers = decoders_share_output_layers
        self._decoders_share_target_embedding = decoders_share_target_embedding
        self._stop_gradient_from_parent_state = stop_gradient_from_parent_state
        self._use_parent_state_gating = use_parent_state_gating
        if use_parent_state_gating:
            self._parent_state_gate = nn.Linear(self.decoder_output_dim, 1)

        if self._child_derivation_use_separate_decoder:
            self._child_derivation_decoder_cell = StackedLSTMCell(
                self.decoder_input_dim, self.decoder_output_dim, num_decoder_layers, decoder_dropout)
            self._child_derivation_decoder_attention = copy.deepcopy(self._attention)
            self._child_derivation_decoder_attention_vec_linear = Linear(
                self.decoder_hidden_dim + self.encoder_output_dim, self.decoder_output_dim, bias=False)

            if not self._decoders_share_output_layers:
                self._child_output_generation_layer = Linear(self.decoder_output_dim, self._target_vocab_size)
                self._child_copy_attention = BilinearAttention(self.decoder_output_dim, self.encoder_output_dim, normalize=False)

            if not self._decoders_share_target_embedding:
                self._child_target_embedder = Embedding(
                    num_embeddings=self._target_vocab_size,
                    embedding_dim=target_embedding_dim
                )
        else:
            self._child_derivation_decoder_cell = lambda *args, **kwargs: self._decoder_cell(*args, **kwargs)
            self._child_derivation_decoder_attention = lambda *args, **kwargs: self._attention(*args, **kwargs)
            self._child_derivation_decoder_attention_vec_linear = (
                lambda *args, **kwargs:
                self._decoder_attention_vec_linear(*args, **kwargs)
            )

        # Inner span selection model over source utterances
        self._derivation_source_span_extractor = EndpointSpanExtractor(
            input_dim=self.encoder_output_dim,
            combination='x,y'
        )
        self._span_representation_linear = nn.Linear(
            self._derivation_source_span_extractor.get_output_dim(),
            target_embedding_dim
        )
        self._span_scorer = BilinearAttention(decoder_hidden_dim, target_embedding_dim, normalize=False)

        # Attention regularization method
        assert attention_regularization is None or attention_regularization.startswith('mse')
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

        if attention_regularization and 'complementary' in attention_regularization and self._normalize_target_attention_distribution:
            raise ValueError(f'conflict config: {attention_regularization}')

        # Parser-specific options
        self._parse_sketch_only = parse_sketch_only

        if root_derivation_compute_span_scores is False:
            if self._parse_sketch_only is False:
                logger.warning(
                    '`root_derivation_compute_span_scores` is turned off, '
                    'but `parse_sketch_only` is off'
                )

                if self._child_derivation_use_root_utterance_encoding is False:
                    raise ConfigurationError(
                        'when `root_derivation_compute_span_scores` is False for hiero model'
                        'child_derivation_use_root_utterance_encoding should be on'
                    )

        self._compute_span_scores_for_root_derivation = root_derivation_compute_span_scores

        if sketch_parser_no_source_span_prediction:
            if child_derivation_compute_span_scores is False:
                raise ConfigurationError(
                    'when `sketch_parser_no_source_span_prediction` is on, '
                    '`child_derivation_compute_span_scores` must remain as default'
                )
            child_derivation_compute_span_scores = False

        self._compute_span_scores_for_child_derivation = child_derivation_compute_span_scores

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_search = BeamSearchWithStatesLogging(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )
        self._force_decoding_sketch = force_decoding_sketch
        self._beam_search_method = beam_search_method
        self._dump_full_beam = dump_full_beam
        self._beam_search_output_attention = beam_search_output_attention

        self._bleu = BLEU(exclude_indices={self._pad_index, self._start_index, self._end_index})
        # self._sequence_accuracy = SequenceAccuracy()
        self._token_based_metric = SequenceCategorizedMatchMetric()
        self._oracle_metric = OracleMetric(SequenceCategorizedMatchMetric())
        self._structured_metric = StructuredRepresentationMetric()

        self._dataset_reader = cast(DecompositionalParserReader, dataset_reader)

        self._debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

        # FIXME: hacky
        initializer._initializers.append(
            ('_decoder_cell_init_linear', Initializer.by_name('zero')())
        )

        initializer(self)

        if kwargs:
            logger.warning(f"The following arguments are not used: {kwargs}")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def source_token_field_key(self):
        return 'token_ids' if self._dataset_reader.use_pretrained_encoder else 'tokens'

    def forward(
        self,
        source_tokens: TextFieldTensors,
        metadata: List[Dict[str, Any]],
        derivation: List[TensorDict] = None,
        # root_derivation_target_to_source_alignment: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        if derivation is not None:
            output_dict = self._forward_loss(
                derivation,
                metadata,
                # root_derivation_target_to_source_alignment
            )
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

        if not self.training:
            predictions = self.inference(
                [meta['source_tokens'] for meta in metadata],
                metadata,
                training_derivation_tensor_dicts=derivation,
                return_attention=self._beam_search_output_attention
            )
            top_predictions: List[Dict] = [hyp_list[0] for hyp_list in predictions["predictions"]]  # noqa

            if self._dump_full_beam:
                output_dict.update(predictions)
            else:
                output_dict['predictions'] = top_predictions

            if 'target_tokens' in metadata[0]:
                hyp_tokens = [hyp['tokens'] for hyp in top_predictions]
                ref_tokens = [meta['target_tokens'] for meta in metadata]
                tags = [x['tags'] for x in metadata]

                if self._token_based_metric:
                    self._token_based_metric(hyp_tokens, ref_tokens, tags)

                    if self._dump_full_beam:
                        self._oracle_metric(
                            [
                                [
                                    hyp['tokens'] for hyp in beam
                                ]
                                for beam in predictions["predictions"]
                            ],
                            ref_tokens, tag_set_lists=tags
                        )

                if self._structured_metric:
                    ref_representations = [
                        seq_to_sexp(ref_sexp_tokens)
                        for ref_sexp_tokens in ref_tokens
                    ]

                    hyp_representations = []
                    for hyp in hyp_tokens:
                        try:
                            hyp_representation = self.token_sequence_to_meaning_representation(hyp)
                        except:
                            logger.error(f"Failed to parse hyp representation {hyp}")
                            hyp_representation = ['INVALID_TREE']

                        hyp_representations.append(hyp_representation)

                    self._structured_metric(
                        hyp_representations,
                        [hyp['derivation']['representation'] for hyp in top_predictions],
                        ref_representations,
                        [x['tags'] for x in metadata]
                    )

        return output_dict

    def _encode(
        self,
        source_tokens: TextFieldTensors,
        source_spans: Optional[torch.LongTensor] = None,
        derivation_level: int = 0
        # source_spans_mask: Optional[torch.BoolTensor] = None
    ) -> StateType:
        """
        Encode source input sentences.
        """
        original_source_tokens = source_tokens
        batch_size, *padding_dims, max_input_sequence_length = source_tokens['tokens'][self.source_token_field_key].size()

        if padding_dims:
            # (batch_size|derivation_level_num|derivation_step_num, max_input_sequence_length)
            source_tokens = utils.flatten_text_field_tensor(
                source_tokens, start_dim=0, end_dim=len(padding_dims))

        source_embedder = self._source_embedder
        encoder = self._encoder
        if derivation_level > 0 and self._child_derivation_use_separate_encoder:
            source_embedder = self._child_source_embedder
            encoder = self._child_encoder

        # shape: (flattened_size, max_input_sequence_length, encoder_input_dim)
        embedded_input = source_embedder(source_tokens)
        # shape: (flattened_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (flattened_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = encoder(embedded_input, source_mask)

        if source_spans is not None:
            if padding_dims:
                # (flattened_size, source_span_num, 2)
                source_spans = source_spans.flatten(0, len(padding_dims))

            # (flattened_size, source_span_num)
            source_spans_mask = source_spans[:, :, 0] != -1
            # (flattened_size, source_span_num, span_size)
            source_spans_encoding = self._derivation_source_span_extractor(
                encoder_outputs, source_spans, span_indices_mask=source_spans_mask)

            source_spans_encoding = self._span_representation_linear(source_spans_encoding)

        if padding_dims:
            encoder_outputs = encoder_outputs.view(*[[batch_size] + padding_dims + [max_input_sequence_length, -1]])
            source_mask = source_mask.view(*[[batch_size] + padding_dims + [max_input_sequence_length]])

            if source_spans is not None:
                source_span_num, span_encoding_size = source_spans_encoding.shape[1:]
                source_spans_encoding = source_spans_encoding.view(
                    *tuple([batch_size] + padding_dims + [source_span_num, span_encoding_size])
                )

                source_spans_mask = source_spans_mask.view(
                    *tuple([batch_size] + padding_dims + [source_span_num])
                )

        original_source_tokens['tokens']['mask'] = source_mask
        output_dict = {"source_tokens_mask": source_mask, "encoder_outputs": encoder_outputs}

        if source_spans is not None:
            output_dict.update({
                'source_spans_encoding': source_spans_encoding,
                'source_spans_mask': source_spans_mask
            })

        return output_dict

    def _init_decoder_state(
        self,
        state: StateType,
        parent_decoder_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> StateType:
        source_tokens_mask = state['source_tokens_mask']

        # Assign the mask tensor of empty source sequence to have one non-zero entry
        # otherwise allennlp will complain.
        source_tokens_mask_clone = source_tokens_mask.clone()
        example_mask = source_tokens_mask.sum(dim=-1) == 0
        source_tokens_mask_clone[example_mask, 0] = True

        # encoder_last_states: (batch_size, encoder_output_dim)
        if self._dataset_reader.use_pretrained_encoder:
            encoder_last_states = state['encoder_outputs'][:, 0]
        else:
            encoder_last_states = util.get_final_encoder_states(
                state['encoder_outputs'],
                source_tokens_mask_clone,
                self._encoder.is_bidirectional()
            )

        batch_size = encoder_last_states.size(0)
        decoder_layer_num = self._decoder_cell.num_layers

        # if it is in root level, we use the last encoder state to initialize the decoder
        if parent_decoder_state is None:
            # shape: (batch_size, decoder_hidden_size)
            decoder_init_state = torch.tanh(
                self._decoder_state_init_linear(encoder_last_states)
                # .view(batch_size, decoder_layer_num, -1)
            )
            # shape: (batch_size, decoder_layer_num, encoder_output_dim)
            decoder_init_state = torch.cat([
                decoder_init_state.unsqueeze(1),
                decoder_init_state.new_zeros(batch_size, decoder_layer_num - 1, self.decoder_hidden_dim)
            ], dim=1)

            decoder_init_cell = decoder_init_state.new_zeros(batch_size, decoder_layer_num, self.decoder_hidden_dim)

            decoder_attentional_vec = torch.zeros(batch_size, self.decoder_hidden_dim, device=self.device)
            # decoder_init_cell = (
            #     self._decoder_cell_init_linear(encoder_last_states)
            #     # .view(batch_size, decoder_layer_num, -1)
            # )
            # decoder_init_cell = torch.cat([
            #     decoder_init_cell.unsqueeze(1),
            #     decoder_init_cell.new_zeros(batch_size, decoder_layer_num - 1, self.decoder_hidden_dim)
            # ], dim=1)
        else:
            # if it is in child layers, we use the parent hidden states and context vector to initialize the decoder,
            # with an additional flag to signal the decoder we are at the child derivation level
            # shape: (batch_size, decoder_layer_num, encoder_output_dim)
            # encoder_last_states = torch.cat([
            #     encoder_last_states.unsqueeze(1),
            #     encoder_last_states.new_zeros(batch_size, decoder_layer_num - 1, self.encoder_output_dim)
            # ], dim=1)

            if self._stop_gradient_from_parent_state and self.training:
                parent_decoder_state = (
                    parent_decoder_state[0].detach(),
                    parent_decoder_state[1].detach()
                )

            # (batch_size, encoder_output_dim)
            parent_att_vec = parent_decoder_state[2]

            if self._use_parent_state_gating:
                # (batch_size, 1)
                gate = torch.sigmoid(
                    self._parent_state_gate(
                        parent_att_vec
                    )
                )

                parent_decoder_state = (
                    parent_decoder_state[0] * gate,
                    parent_decoder_state[1] * gate
                )

            # we duplicate the context vector, and dispatch it to
            # (batch_size, decoder_layer_num, hidden_size)
            parent_hidden_state = parent_decoder_state[0]
            parent_cell = parent_decoder_state[1]

            decoder_init_state_input_vecs = [
                parent_hidden_state,
                self._child_level_decoder_start_tag.unsqueeze(0).unsqueeze(0).expand(batch_size, decoder_layer_num, -1)
            ]

            decoder_init_cell_input_vecs = [
                parent_cell,
                self._child_level_decoder_start_tag.unsqueeze(0).unsqueeze(0).expand(batch_size, decoder_layer_num, -1)
            ]

            if not self._child_derivation_use_root_utterance_encoding and not self._parse_sketch_only:
                expanded_encoder_last_states = torch.cat([
                    encoder_last_states.unsqueeze(1),
                    torch.zeros(batch_size, decoder_layer_num - 1, encoder_last_states.size(-1), device=self.device)
                ], dim=1)
                decoder_init_state_input_vecs.append(expanded_encoder_last_states)
                decoder_init_cell_input_vecs.append(expanded_encoder_last_states)

            decoder_init_state = torch.tanh(
                self._decoder_state_with_parent_init_linear_layer(
                    torch.cat(decoder_init_state_input_vecs, dim=-1)
                )
            )
            decoder_init_cell = self._decoder_cell_with_parent_init_linear_layer(
                torch.cat(decoder_init_cell_input_vecs, dim=-1)
            )

            decoder_attentional_vec = parent_att_vec

        state.update({
            'decoder_hidden': decoder_init_state,
            'decoder_cell': decoder_init_cell,
            'decoder_attentional_vec': decoder_attentional_vec
        })

        return state

    def _forward_loss(
        self,
        derivation: List[TensorDict],
        metadata: List[Dict[str, Any]],
        # root_derivation_target_to_source_alignment: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size = derivation[0]['derivation_step_target_tokens']['tokens']['tokens'].size(0)

        # (batch_size, derivation_step_num)
        derivation_forward_output = self._forward_derivation(derivation, metadata)

        # shape: (batch_size, total_flattened_derivation_step_num)
        tgt_derivation_step_log_probs = derivation_forward_output['target_derivation_step_log_probs']

        # Sum of step log-likelihoods.
        if self._parse_sketch_only:
            # shape: (batch_size, )
            tgt_log_likelihood = tgt_derivation_step_log_probs[:, 0]
        else:
            # shape: (num_decomposed_source,)
            tgt_log_likelihood = tgt_derivation_step_log_probs.sum()

        # The loss is the negative log-likelihood, averaged over the batch.
        loss = -tgt_log_likelihood.sum() / batch_size

        output_dict = {}

        # apply attention regularization
        if self._attention_regularization and self.training:
            att_reg_loss = 0.
            for derivation_level, deriv_level_tensor_dict in enumerate(derivation):
                deriv_step_target_to_source_alignment = deriv_level_tensor_dict.get(
                    'derivation_step_target_to_source_alignment')

                if (
                    deriv_step_target_to_source_alignment is not None and
                    not (
                        derivation_level > 0 and
                        (
                            self._sketch_level_attention_regularization_only or
                            self._parse_sketch_only
                        )
                    )
                ):
                    # shape: (batch_size, derivation_step_num, target_sequence_length - 1, source_sequence_length)
                    derivation_step_attention_weights = derivation_forward_output['target_derivation_step_attention_weights'][derivation_level]

                    # shape: (batch_size, derivation_step_num, target_sequence_length - 1, source_sequence_length)
                    deriv_step_target_to_source_alignment = deriv_step_target_to_source_alignment[:, :, 1:]
                    deriv_step_target_to_source_alignment_mask = (deriv_step_target_to_source_alignment != -1)
                    deriv_step_target_to_source_alignment = deriv_step_target_to_source_alignment * deriv_step_target_to_source_alignment_mask

                    if derivation_level > 0:
                        valid_source_token_num = deriv_step_target_to_source_alignment.size(-1)
                        if valid_source_token_num != derivation_step_attention_weights.size(-1):
                            derivation_step_attention_weights = derivation_step_attention_weights.narrow(
                                dim=-1, start=0, length=valid_source_token_num)

                    # shape: (batch_size, derivation_step_num, target_sequence_length - 1, source_sequence_length)
                    target_attention_distribution = deriv_step_target_to_source_alignment
                    target_attention_distribution_mask = deriv_step_target_to_source_alignment_mask
                    # shape: (batch_size, derivation_step_num, target_sequence_length - 1)
                    target_attention_distribution_target_timestep_mask = target_attention_distribution_mask.any(dim=-1)

                    if self._normalize_target_attention_distribution:
                        # shape: (batch_size, derivation_step_num, target_sequence_length - 1, source_sequence_length)
                        target_attention_distribution = target_attention_distribution / (
                            target_attention_distribution.sum(dim=-1) +
                            util.tiny_value_of_dtype(
                                torch.float) * ~target_attention_distribution_target_timestep_mask
                        ).unsqueeze(-1)

                    # shape: (batch_size * derivation_step_num, target_sequence_length - 1, source_sequence_length)
                    deriv_step_att_reg = nn.MSELoss(reduction='none')(
                        derivation_step_attention_weights.flatten(0, 1),
                        target_attention_distribution.flatten(0, 1)
                    ) * target_attention_distribution_mask.flatten(0, 1)

                    # shape: (batch_size, derivation_step_num, target_sequence_length)
                    deriv_level_tgt_token_mask = deriv_level_tensor_dict['derivation_step_target_tokens']['tokens']['tokens'] != 0
                    # shape: (batch_size, derivation_step_num, source_sequence_length)
                    deriv_level_src_token_mask = deriv_level_tensor_dict['derivation_step_source_tokens']['tokens']['mask']
                    # shape: (batch_size, derivation_step_num, 1)
                    num_attn_regularized_src_tokens = deriv_level_src_token_mask.sum(dim=-1, keepdim=True)
                    # shape: (batch_size, derivation_step_num)
                    num_attn_regularized_tgt_tokens = (deriv_step_target_to_source_alignment.sum(dim=-1) > 0).sum(dim=-1)

                    # shape: (batch_size * derivation_step_num)
                    deriv_step_att_reg = (
                        deriv_step_att_reg.sum(dim=-1)
                        # /
                        # (
                        #     num_attn_regularized_src_tokens.flatten(0, 1) +
                        #     util.tiny_value_of_dtype(torch.float)
                        # )
                    ).sum(dim=-1) / (
                        num_attn_regularized_tgt_tokens.flatten(0, 1) +
                        util.tiny_value_of_dtype(torch.float)
                    )

                    level_valid_derivation_step_num = deriv_level_tgt_token_mask.any(dim=-1).sum()
                    deriv_step_att_reg = deriv_step_att_reg.sum() / level_valid_derivation_step_num
                    att_reg_loss = att_reg_loss + deriv_step_att_reg

            att_reg_loss = att_reg_loss * self._attention_reg_weight

            loss = loss + att_reg_loss

        output_dict = {
            'loss': loss,
        }

        return output_dict

    def _forward_derivation(
        self,
        derivation: DerivationTensorDictList,
        metadata: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            derivation: List of tensor dicts with the following keys:
                * derivation_step_source_tokens:
                    Shape (batch_size, derivation_level_num, derivation_step_num, source_sequence_length)
                * derivation_step_source_token_first_appearing_indices:
                    Shape (batch_size, derivation_level_num, derivation_step_num, source_sequence_length)
                * derivation_step_source_spans:
                    Shape (batch_size, derivation_level_num, derivation_step_num, source_span_num, [src_span_start_idx, src_span_end_idx])
                * derivation_step_target_tokens:
                    Shape (batch_size, derivation_level_num, derivation_step_num, target_sequence_length)
                # derivation_step_parent_tgt_marker_timestep:
                #     Shape (batch_size, derivation_level_num, derivation_step_num, [parent_derivation_step_id, timestep_offset])
                * new_derivation_step_marker_source_span_id:
                    Shape (batch_size, derivation_level_num, derivation_step_num, target_sequence_length)
                * derivation_step_target_token_first_appearing_indices:
                    Shape (batch_size, derivation_level_num, derivation_step_num, target_sequence_length)
        """

        batch_size = derivation[0]['derivation_step_source_tokens']['tokens'][self.source_token_field_key].size(0)
        derivation_level_num = len(derivation)

        parent_decoder_state = None
        derivation_states: List[StateType] = []
        target_derivation_step_log_probs_list: List[torch.Tensor] = []
        attention_weights_list: List[torch.Tensor] = []

        if self._parse_sketch_only:
            derivation_level_num = 1

        for deriv_level in range(derivation_level_num):
            deriv_level_tensor_dict = derivation[deriv_level]

            derivation_step_num = deriv_level_tensor_dict['derivation_step_target_tokens']['tokens']['tokens'].size(1)

            # flatten the tensors
            deriv_step_source_tokens = utils.flatten_text_field_tensor(
                deriv_level_tensor_dict['derivation_step_source_tokens'],
                0, 1
            )

            deriv_step_target_tokens = utils.flatten_text_field_tensor(
                deriv_level_tensor_dict['derivation_step_target_tokens'],
                0, 1
            )

            deriv_step_target_tokens['tokens']['mask'] = util.get_text_field_mask(deriv_step_target_tokens)

            deriv_step_source_spans = deriv_level_tensor_dict['derivation_step_source_spans'].flatten(0, 1)

            deriv_step_source_token_first_appearing_indices = (
                deriv_level_tensor_dict['derivation_step_source_token_first_appearing_indices']
                .flatten(0, 1)
            )

            deriv_step_target_token_first_appearing_indices = (
                deriv_level_tensor_dict['derivation_step_target_token_first_appearing_indices']
                .flatten(0, 1)
            )

            deriv_step_target_token_source_span_id = (
                deriv_level_tensor_dict['derivation_step_target_token_source_span_id']
                .flatten(0, 1)
            )

            deriv_step_new_deriv_marker_mask = deriv_step_target_tokens['tokens']['tokens'] == self._child_derivation_step_marker_idx
            deriv_step_target_token_source_span_id_mask = deriv_step_target_token_source_span_id != -1
            deriv_step_target_token_source_span_id.masked_fill_(~deriv_step_target_token_source_span_id_mask, 0)

            if deriv_level > 0 and self._child_derivation_use_root_utterance_encoding:
                child_deriv_step_root_example_id = [0] * (batch_size * derivation_step_num)
                parent_source_spans = [None] * (batch_size * derivation_step_num)

                for e_id in range(batch_size):
                    child_derivation_steps: List[Dict] = metadata[e_id]['derivation']['child_derivation_steps']
                    for deriv_step_id, deriv_step in enumerate(child_derivation_steps):
                        child_deriv_step_root_example_id[e_id * derivation_step_num + deriv_step_id] = e_id
                        if not deriv_step['is_floating']:
                            parent_source_spans[e_id * derivation_step_num + deriv_step_id] = list(deriv_step['source_span_position'])

                for i in range(len(parent_source_spans)):
                    if parent_source_spans[i] is None:
                        parent_source_spans[i] = [0, 0]

                # deriv_step_source_tokens = utils.text_field_tensor_apply(
                #     derivation[0]['derivation_step_source_tokens'],
                #     lambda tensor: tensor[child_deriv_step_root_example_id].flatten(0, 1)
                # )
                # deriv_step_source_token_first_appearing_indices = (
                #     derivation[0]['derivation_step_source_token_first_appearing_indices']
                #     [child_deriv_step_root_example_id]
                #     .flatten(0, 1)
                # )

                root_state = derivation_states[0]
                state = {
                    'encoder_outputs': root_state['encoder_outputs'][child_deriv_step_root_example_id],
                    'source_tokens_mask': root_state['source_tokens_mask'][child_deriv_step_root_example_id],
                    'parent_source_spans': parent_source_spans
                }

                root_max_source_token_length = state['encoder_outputs'].size(1)
                deriv_step_max_source_token_length = deriv_step_source_tokens['tokens'][self.source_token_field_key].size(1)

                if deriv_step_max_source_token_length < root_max_source_token_length:
                    state['encoder_outputs'] = torch.narrow(
                        state['encoder_outputs'],
                        dim=1,
                        start=0, length=deriv_step_max_source_token_length
                    )

                    state['source_tokens_mask'] = torch.narrow(
                        state['source_tokens_mask'],
                        dim=1,
                        start=0, length=deriv_step_max_source_token_length
                    )
            else:
                state = self._encode(
                    deriv_step_source_tokens,
                    source_spans=deriv_step_source_spans,
                    derivation_level=deriv_level
                )

            if parent_decoder_state is not None:
                parent_decoder_state = (
                    parent_decoder_state[0].flatten(0, 1),
                    parent_decoder_state[1].flatten(0, 1),
                    parent_decoder_state[2].flatten(0, 1),
                )

            if self._debug:
                if deriv_level == 0:
                    for e_id in range(batch_size):
                        child_derivation_steps: List[Dict] = metadata[e_id]['derivation']['child_derivation_steps']
                        for deriv_step_id, deriv_step in enumerate(child_derivation_steps):
                            src_span_start, src_span_end = deriv_step['source_span_position']
                            span_idx = deriv_step['source_span_slice_idx']
                            assert deriv_step_source_spans[
                                e_id,
                                span_idx
                            ].detach().cpu().tolist() == [src_span_start, src_span_end - 1]  # inclusive indexing
                            assert deriv_step_target_token_source_span_id[
                                e_id,
                                deriv_step['target_span_position'] + 1  # account for the leading <s> symbol
                            ].item() == span_idx
                            assert deriv_step_target_tokens['tokens']['tokens'][
                                e_id,
                                deriv_step['target_span_position'] + 1
                            ].item() == self._child_derivation_step_marker_idx
                else:
                    for e_id in range(batch_size):
                        child_derivation_steps: List[Dict] = metadata[e_id]['derivation']['child_derivation_steps']
                        for deriv_step_id, deriv_step in enumerate(child_derivation_steps):
                            if self._child_derivation_use_root_utterance_encoding:
                                root_source_tokens = (
                                    derivation[0]['derivation_step_source_tokens']
                                    ['tokens'][self.source_token_field_key]
                                    [e_id, 0]
                                )
                                _deriv_step_source_tokens = (
                                    derivation[1]['derivation_step_source_tokens']
                                    ['tokens'][self.source_token_field_key]
                                    [e_id, deriv_step_id]
                                )

                                assert torch.all(
                                    root_source_tokens[:_deriv_step_source_tokens.size(0)] ==
                                    _deriv_step_source_tokens
                                )
                            else:
                                src_span_start, src_span_end = deriv_step['source_span_position']
                                span_length = src_span_end - src_span_start
                                deriv_step_token_id_list: List[int] = (
                                    deriv_step_source_tokens['tokens'][self.source_token_field_key]
                                    [e_id * derivation_step_num + deriv_step_id, :span_length]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )
                                deriv_step_root_src_span_token_id_list: List[int] = (
                                    derivation[0]['derivation_step_source_tokens']
                                    ['tokens'][self.source_token_field_key]
                                    [e_id, 0, src_span_start: src_span_end]
                                    .detach()
                                    .cpu()
                                    .tolist()
                                )

                                assert deriv_step_token_id_list == deriv_step_root_src_span_token_id_list

            deriv_level_decode_output = self._forward_derivation_step_decode(
                deriv_step_target_tokens,
                deriv_step_target_token_first_appearing_indices,
                deriv_step_source_token_first_appearing_indices,
                deriv_step_new_deriv_marker_mask[:, 1:],
                deriv_step_target_token_source_span_id[:, 1:],
                deriv_step_target_token_source_span_id_mask[:, 1:],
                parent_decoder_state,
                batch_size=batch_size,
                derivation_step_num=derivation_step_num,
                derivation_level=deriv_level,
                state=state
            )

            # shape: (batch_size * derivation_step_num, target_sequence_length - 1)
            deriv_step_tgt_seq_log_probs = self._get_forward_log_likelihoods(
                source_token_mask=deriv_step_source_tokens['tokens']['mask'],
                source_spans_mask=state.get('source_spans_mask'),
                target_tokens_id=deriv_step_target_tokens['tokens']['tokens'][:, 1:],
                target_tokens_mask=deriv_step_target_tokens['tokens']['mask'][:, 1:],
                **deriv_level_decode_output,
                derivation_level=deriv_level
            )

            # (batch_size, next_level_derivation_step_num, decoder_layer_num, decoder_hidden_size)
            parent_decoder_hidden = deriv_level_decode_output['new_deriv_step_decoder_hidden']
            parent_decoder_cell = deriv_level_decode_output['new_deriv_step_decoder_cell']
            parent_decoder_att_vec = deriv_level_decode_output['new_deriv_step_decoder_attentional_vec']
            # (batch_size, next_level_derivation_step_num, decoder_layer_num, decoder_hidden_size)
            parent_decoder_state = (parent_decoder_hidden, parent_decoder_cell, parent_decoder_att_vec)

            target_derivation_step_log_probs_list.append(
                deriv_step_tgt_seq_log_probs.sum(dim=-1).view(batch_size, derivation_step_num)
            )

            deriv_level_attn_weights = deriv_level_decode_output['attention_weights']
            deriv_level_attn_weights = deriv_level_attn_weights.view(
                *((batch_size, derivation_step_num, ) + deriv_level_attn_weights.shape[1:]))
            attention_weights_list.append(deriv_level_attn_weights)

            derivation_states.append(state)

        # (batch_size, total_flattened_derivation_step_num)
        tgt_deriv_step_log_probs = torch.cat(target_derivation_step_log_probs_list, dim=1)

        return {
            'target_derivation_step_log_probs': tgt_deriv_step_log_probs,
            'target_derivation_step_attention_weights': attention_weights_list
        }

    def _forward_derivation_step_decode(
        self,
        target_tokens: LogicalFormTensors,
        target_token_first_appearing_indices: torch.LongTensor,
        source_token_first_appearing_indices: torch.LongTensor,
        new_derivation_step_marker_mask: torch.BoolTensor,
        new_derivation_step_marker_source_span_id: torch.LongTensor,
        new_derivation_step_marker_source_span_id_mask: torch.LongTensor,
        parent_decoder_state: torch.Tensor,
        batch_size: int,
        derivation_step_num: int,
        derivation_level: int,
        state: StateType
    ):
        """
        Args:
            Input tensors have shape of (batch_size * derivation_step_num, ...)
            parent_decoder_state:
                Shape (batch_size * derivation_step_num, decoder_layer_num, decoder_output_size)
        """

        total_derivation_step_num, target_sequence_length = target_tokens['tokens']['tokens'].size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        # We use this to fill in the copy index when the previous input was copied.
        # shape: (total_derivation_step_num,)
        copy_symbols = cast(torch.LongTensor, torch.full(
            (total_derivation_step_num, 1), fill_value=self._copy_index, dtype=torch.long, device=self.device
        ))

        # shape: (total_derivation_step_num, target_sequence_length, source_sequence_length)
        target_token_source_position_mask = (
            (
                source_token_first_appearing_indices.unsqueeze(1) ==
                target_token_first_appearing_indices.unsqueeze(-1)
            ) &
            target_tokens['tokens']['mask'].unsqueeze(-1)
        )

        # shape: (total_derivation_step_num, target_sequence_length)
        target_token_copiable_mask = target_token_source_position_mask.sum(dim=-1) > 0

        # shape: (total_derivation_step_num, target_sequence_length)
        target_tokens_id = target_tokens["tokens"]["tokens"]

        # shape: (total_derivation_step_num, target_sequence_length)
        is_target_token_copied_only = (
            (target_tokens_id == self._oov_index) & target_token_copiable_mask  # noqa
        ).long()

        # shape: (total_derivation_step_num, target_sequence_length)
        target_token_symbols = (
            is_target_token_copied_only * copy_symbols +
            (1 - is_target_token_copied_only) * target_tokens_id
        )

        compute_source_span_scores: bool = self._compute_source_span_scores_for_derivation_level(derivation_level)

        # shape: (total_derivation_step_num, target_sequence_length, embedding_size)
        target_token_embeds = self._get_target_embedder(derivation_level)(target_token_symbols)

        # Initialize decoder
        self._init_decoder_state(state, parent_decoder_state)

        # Initialize attention vector
        att_tm1 = state['decoder_attentional_vec']

        att_vecs = []
        att_weights = []
        copy_scores = []
        new_deriv_step_src_span_scores = []
        new_deriv_marker_target_timesteps = []
        new_deriv_marker_decoder_states: List[List[Tuple]] = [list() for _ in range(total_derivation_step_num)]
        tm1_use_src_span_selection_mask = None
        prev_step_has_span_selection = False

        target_token_embeds_list: List[torch.Tensor] = torch.split(target_token_embeds, 1, dim=1)

        if compute_source_span_scores:
            # shape: (total_derivation_step_num, source_span_num)
            source_spans_encoding = state['source_spans_encoding']

        new_derivation_step_marker_mask_list: List[torch.Tensor] = torch.split(
            new_derivation_step_marker_mask, 1, dim=1
        )

        target_step_has_deriv_marker_list: List[torch.BoolTensor] = torch.split(
            torch.any(new_derivation_step_marker_mask, dim=0),  # (target_sequence_length,)
            1,
            dim=0
        )

        for time_step in range(num_decoding_steps):
            # shape: (batch_size, target_embedding_size)
            y_tm1_embed = target_token_embeds_list[time_step].squeeze(1)

            # if (
            #     compute_source_span_scores and
            #     tm1_use_src_span_selection_mask is not None and
            #     prev_step_has_span_selection
            # ):
            #     # (batch_size,)
            #     source_span_id_tm1 = new_derivation_step_marker_source_span_id[:, time_step - 1]
            #     # (batch_size, target_embedding_dim)
            #     src_span_encoding_tm1 = source_spans_encoding.gather(
            #         dim=1,
            #         index=source_span_id_tm1.view(-1, 1, 1).clone().expand(-1, -1, source_spans_encoding.size(-1))
            #     ).squeeze(1)
            #
            #     tm1_use_src_span_selection_mask = tm1_use_src_span_selection_mask.unsqueeze(-1)
            #     y_tm1_embed = (
            #         y_tm1_embed * ~tm1_use_src_span_selection_mask +
            #         src_span_encoding_tm1 * tm1_use_src_span_selection_mask
            #     )

            # shape: (batch_size, target_embedding_dim + decoder_hidden_dim)
            x_t = torch.cat([y_tm1_embed, att_tm1], dim=-1)

            # Update the decoder state by taking a step through the RNN.
            # `decoder_output` shape: (batch_size, decoder_hidden_size)
            # `att_weight_t` shape: (batch_size, source_input_length)
            att_t, state, att_weight_t = self._decoder_step(
                x_t, state,
                return_attention=True, derivation_level=derivation_level
            )

            step_copy_scores = self._get_copy_scores(att_t, state, derivation_level=derivation_level)

            att_vecs.append(att_t)
            att_weights.append(att_weight_t)
            copy_scores.append(step_copy_scores)

            if compute_source_span_scores:
                # (batch_size, source_span_num)
                new_deriv_step_source_span_scores_t = self._get_new_derivation_step_source_span_probs(
                    att_t, state, normalize=False
                )
                new_deriv_step_src_span_scores.append(new_deriv_step_source_span_scores_t)

            # (batch_size,)
            new_deriv_step_marker_mask_t = new_derivation_step_marker_mask_list[time_step].squeeze(1)
            step_target_has_deriv_marker = target_step_has_deriv_marker_list[time_step].item()

            if step_target_has_deriv_marker:
                new_deriv_step_marker_mask_t.detach().cpu().tolist()
                for example_idx in [
                    idx
                    for idx, mask
                    in enumerate(new_deriv_step_marker_mask_t)
                    if mask
                ]:
                    # Shape: (decoder_layer_num, hidden_size)
                    example_state = (
                        state['decoder_hidden'][example_idx],
                        state['decoder_cell'][example_idx],
                        att_t[example_idx]
                    )
                    new_deriv_marker_decoder_states[example_idx].append(example_state)

                prev_step_has_span_selection = True
                new_deriv_marker_target_timesteps.append(time_step)
            else:
                prev_step_has_span_selection = False

            tm1_use_src_span_selection_mask = new_deriv_step_marker_mask_t

            att_tm1 = att_t

        # shape: (batch_size, target_sequence_length - 1, decoder_hidden_size)
        decoder_att_vec = torch.stack(att_vecs, dim=1)

        # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
        att_weights = torch.stack(att_weights, dim=1)

        # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
        copy_scores = torch.stack(copy_scores, dim=1)

        # shape: (batch_size, target_sequence_length - 1, target_vocab_size)
        generation_scores = self._get_generation_scores(decoder_att_vec, derivation_level=derivation_level)

        if compute_source_span_scores:
            # shape: (batch_size, target_sequence_length - 1, source_span_num)
            new_deriv_step_src_span_scores = torch.stack(new_deriv_step_src_span_scores, dim=1)
        else:
            new_deriv_step_src_span_scores = None

        if new_deriv_marker_target_timesteps:
            new_deriv_marker_decoder_cell, new_deriv_marker_decoder_hidden, new_deriv_marker_decoder_att_vec = \
                self._gather_child_derivation_step_parent_decoder_states(
                    new_deriv_marker_decoder_states,
                    batch_size=batch_size, derivation_step_num=derivation_step_num
                )
        else:
            new_deriv_marker_decoder_hidden = new_deriv_marker_decoder_cell = new_deriv_marker_decoder_att_vec = None

        return {
            'attention_weights': att_weights,
            'copy_scores': copy_scores,
            'generation_scores': generation_scores,
            'target_token_source_position_mask': target_token_source_position_mask[:, 1:],
            'new_deriv_step_marker_mask': new_derivation_step_marker_mask,
            'new_deriv_step_source_span_scores': new_deriv_step_src_span_scores,
            'new_deriv_step_source_span_id': new_derivation_step_marker_source_span_id,
            'new_deriv_step_source_span_id_mask': new_derivation_step_marker_source_span_id_mask,
            'new_deriv_step_marker_target_timesteps': new_deriv_marker_target_timesteps,
            'new_deriv_step_decoder_hidden': new_deriv_marker_decoder_hidden,
            'new_deriv_step_decoder_cell': new_deriv_marker_decoder_cell,
            'new_deriv_step_decoder_attentional_vec': new_deriv_marker_decoder_att_vec
        }

    def _gather_child_derivation_step_parent_decoder_states(
        self,
        new_deriv_marker_decoder_states: List[List[Tuple[torch.Tensor, torch.Tensor]]],
        *,
        batch_size: int,
        derivation_step_num: int
    ):
        # gather the number of new child derivation steps per example in batch
        new_deriv_step_num_list: List[int] = []
        for batch_example_idx in range(batch_size):
            new_deriv_step_num = sum(
                len(states)
                for states
                in new_deriv_marker_decoder_states[
                   batch_example_idx * derivation_step_num:
                   (batch_example_idx + 1) * derivation_step_num]
            )
            new_deriv_step_num_list.append(new_deriv_step_num)
        max_new_deriv_step_num = max(new_deriv_step_num_list)
        flattened_decoder_hidden = []
        flattened_decoder_cell = []
        flattened_decoder_att_vec = []
        flattened_state_idx_list = []
        for batch_example_idx in range(batch_size):
            new_deriv_marker_id = 0
            for states in new_deriv_marker_decoder_states[
                batch_example_idx * derivation_step_num:
                (batch_example_idx + 1) * derivation_step_num
            ]:
                for state in states:
                    flattened_state_idx = batch_example_idx * max_new_deriv_step_num + new_deriv_marker_id
                    flattened_state_idx_list.append(flattened_state_idx)
                    flattened_decoder_hidden.append(state[0])
                    flattened_decoder_cell.append(state[1])
                    flattened_decoder_att_vec.append(state[2])

                    new_deriv_marker_id += 1

        # (total_new_deriv_state_num)
        flattened_state_idx_list = torch.tensor(flattened_state_idx_list, device=self.device)
        # (total_new_deriv_state_num, decoder_layer_num, decoder_hidden_size)
        flattened_state_idx_list = (
            flattened_state_idx_list
            .view(-1, 1, 1)
            .expand(-1, self._decoder_cell.num_layers, self.decoder_hidden_dim)
        )

        # (total_new_deriv_state_num, decoder_layer_num, decoder_hidden_size)
        flattened_decoder_hidden = torch.stack(flattened_decoder_hidden, dim=0)
        new_deriv_marker_decoder_hidden = torch.zeros(
            batch_size * max_new_deriv_step_num,
            self._decoder_cell.num_layers,
            self.decoder_hidden_dim,
            device=self.device
        )
        new_deriv_marker_decoder_hidden.scatter_(
            dim=0,
            index=flattened_state_idx_list,
            src=flattened_decoder_hidden
        )
        # (batch_size * derivation_step_num, max_new_deriv_step_num, decoder_layer_num, decoder_hidden_size)
        new_deriv_marker_decoder_hidden = new_deriv_marker_decoder_hidden.view(
            batch_size, max_new_deriv_step_num,
            self._decoder_cell.num_layers, self.decoder_hidden_dim)

        flattened_decoder_cell = torch.stack(flattened_decoder_cell, dim=0)
        new_deriv_marker_decoder_cell = torch.zeros(
            batch_size * max_new_deriv_step_num,
            self._decoder_cell.num_layers,
            self.decoder_hidden_dim,
            device=self.device
        )
        new_deriv_marker_decoder_cell.scatter_(
            dim=0,
            index=flattened_state_idx_list,
            src=flattened_decoder_cell
        )
        new_deriv_marker_decoder_cell = new_deriv_marker_decoder_cell.view(
            batch_size, max_new_deriv_step_num,
            self._decoder_cell.num_layers, self.decoder_hidden_dim)

        flattened_decoder_att_vec = torch.stack(flattened_decoder_att_vec, dim=0)
        new_deriv_marker_att_vec = torch.zeros(
            batch_size * max_new_deriv_step_num,
            self.decoder_hidden_dim,
            device=self.device
        )
        new_deriv_marker_att_vec.scatter_(
            dim=0,
            index=flattened_state_idx_list[:, 0],
            src=flattened_decoder_att_vec
        )
        new_deriv_marker_att_vec = new_deriv_marker_att_vec.view(
            batch_size, max_new_deriv_step_num,
            self.decoder_hidden_dim
        )

        return new_deriv_marker_decoder_cell, new_deriv_marker_decoder_hidden, new_deriv_marker_att_vec

    def _decoder_step(
        self,
        x_t: torch.Tensor,
        state_tm1: Dict[str, torch.Tensor],
        return_attention: bool = False,
        derivation_level: int = 0
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]
    ]:
        if derivation_level == 0:
            attention = self._attention
            decoder_cell = self._decoder_cell
            decoder_attention_vec_linear = self._decoder_attention_vec_linear
        else:
            attention = self._child_derivation_decoder_attention
            decoder_cell = self._child_derivation_decoder_cell
            decoder_attention_vec_linear = self._child_derivation_decoder_attention_vec_linear

        decoder_output_t, (h_t, cell_t) = decoder_cell(
            x_t, (state_tm1["decoder_hidden"], state_tm1["decoder_cell"])
        )

        args = []
        if isinstance(attention, SpanMaskingAttention):
            args = [state_tm1.get('parent_source_spans')]

        # shape: (batch_size, max_input_sequence_length)
        attention_weights = attention(
            decoder_output_t, state_tm1["encoder_outputs"],
            state_tm1["source_tokens_mask"],
            *args
        )

        # shape: (batch_size, encoder_output_dim)
        context_vec = util.weighted_sum(state_tm1["encoder_outputs"], attention_weights)

        # shape: (batch_size, decoder_output_dim)
        attentional_vec = self._decoder_dropout(torch.tanh(
            decoder_attention_vec_linear(torch.cat([decoder_output_t, context_vec], dim=-1))
        ))

        state_tm1.update({
            'decoder_hidden': h_t,
            'decoder_cell': cell_t,
            'decoder_attentional_vec': attentional_vec
        })

        state_t = state_tm1

        return_tuple = (attentional_vec, state_t)
        if return_attention:
            return_tuple = return_tuple + (attention_weights, )

        return return_tuple

    def _get_generation_scores(self, att_vec: torch.Tensor, derivation_level: int = 0) -> torch.Tensor:
        output_layer = (
            self._output_generation_layer
            if derivation_level == 0 or self._decoders_share_output_layers
            else self._child_output_generation_layer
        )

        return output_layer(att_vec)

    def _get_copy_scores(self, att_vec: torch.Tensor, state: Dict[str, torch.Tensor], derivation_level: int = 0) -> torch.Tensor:
        output_layer = (
            self._copy_attention
            if derivation_level == 0 or self._decoders_share_output_layers
            else self._child_copy_attention
        )

        copy_scores = output_layer(att_vec, state['encoder_outputs'])

        return copy_scores

    def _get_target_embedder(self, derivation_level: int = 0) -> nn.Module:
        if derivation_level == 0 or self._decoders_share_target_embedding:
            return self._target_embedder
        else:
            return self._child_target_embedder

    def _compute_source_span_scores_for_derivation_level(
        self,
        derivation_level: int,
    ):
        compute_source_span_scores = (
            (
                derivation_level == 0 and
                self._compute_span_scores_for_root_derivation
            ) or (
                derivation_level > 0 and
                self._compute_span_scores_for_child_derivation
            )
        )

        return compute_source_span_scores

    def _get_new_derivation_step_source_span_probs(
        self,
        att_vec: torch.Tensor,
        state: StateType,
        normalize: bool = True
    ) -> torch.Tensor:
        child_deriv_span_scores = self._span_scorer(
            vector=att_vec,
            matrix=state['source_spans_encoding']
        )

        if normalize:
            child_deriv_span_scores = util.masked_log_softmax(child_deriv_span_scores, mask=state['source_spans_mask'])

        return child_deriv_span_scores

    def _get_forward_log_likelihoods(
        self,
        generation_scores: torch.Tensor,
        copy_scores: torch.Tensor,
        new_deriv_step_marker_mask: torch.BoolTensor,
        new_deriv_step_source_span_scores: torch.Tensor,
        new_deriv_step_source_span_id: torch.LongTensor,
        new_deriv_step_source_span_id_mask: torch.BoolTensor,
        new_deriv_step_marker_target_timesteps: List,
        source_token_mask: torch.BoolTensor,
        source_spans_mask: torch.BoolTensor,
        target_tokens_id: torch.LongTensor,
        target_tokens_mask: torch.BoolTensor,
        target_token_source_position_mask: torch.BoolTensor,
        derivation_level: int = 0,
        **kwargs
    ) -> torch.Tensor:
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

        compute_source_span_scores = self._compute_source_span_scores_for_derivation_level(derivation_level)

        generation_scores_mask = torch.full(
            (batch_size, 1, target_vocab_size),
            fill_value=True, dtype=torch.bool, device=self.device
        )

        scores_masks_list = [generation_scores_mask, source_token_mask.unsqueeze(1)]
        scores_list = [generation_scores, copy_scores]

        if compute_source_span_scores:
            scores_masks_list.append(source_spans_mask.unsqueeze(1))
            scores_list.append(new_deriv_step_source_span_scores)

        # shape: (batch_size, target_sequence_length - 1, target_vocab_size + source_sequence_length + source_spans_mask)  # noqa
        scores_mask = torch.cat(
            scores_masks_list, dim=-1
        ).expand(-1, num_decoding_time_step, -1)

        source_spans_offset = target_vocab_size + source_token_mask.size(-1)

        # shape: (batch_size, target_sequence_length - 1, target_vocab_size + source_sequence_length + source_spans_num)
        all_scores = torch.cat(scores_list, dim=-1)

        # Globally normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_sequence_length + source_spans_num)
        log_probs = util.masked_log_softmax(all_scores, mask=scores_mask, dim=-1)

        # Calculate the log probability for copying each token in the source sentence
        # that matches the current target token. We use the sum of these copy probabilities
        # for matching tokens in the source sentence to get the total probability
        # for the target token.
        # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
        tgt_copy_log_probs = (
            log_probs[:, :, target_vocab_size: source_spans_offset]
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

        # (batch_size, target_sequence_length - 1)
        target_tokens_id_with_span_idx = target_tokens_id
        if (
            compute_source_span_scores and
            new_deriv_step_marker_target_timesteps
        ):
            target_tokens_id_with_span_idx = target_tokens_id.clone()
            target_tokens_id_with_span_idx[new_deriv_step_source_span_id_mask] = (
                new_deriv_step_source_span_id[new_deriv_step_source_span_id_mask] + source_spans_offset
            )

            # we don't want to maximize the probability of derivation marker tokens like `__SLOT{%d}__`
            # they are used as input to decoder only
            target_tokens_mask[new_deriv_step_marker_mask & ~new_deriv_step_source_span_id_mask] = False

        # Now we get the generation score for the gold target token.
        # shape: (batch_size, target_sequence_length - 1, 1)
        tgt_generation_log_probs = log_probs.gather(dim=-1, index=target_tokens_id_with_span_idx.unsqueeze(-1)) + log_generation_mask

        # ... and add the copy score to get the step log likelihood.
        # shape: (batch_size, target_sequence_length - 1, 1 + source_sequence_length)
        combined_gen_and_copy = torch.cat((tgt_generation_log_probs, tgt_copy_log_probs), dim=-1)

        # shape: (batch_size, target_sequence_length - 1)
        sequence_log_probs = util.logsumexp(combined_gen_and_copy, dim=-1)

        sequence_log_probs = sequence_log_probs * target_tokens_mask

        return sequence_log_probs

    def inference(
        self,
        batched_source_tokens: List[List[str]],
        metadata: List[Dict[str, Any]] = None,
        training_derivation_tensor_dicts: List[StateType] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Perform beam search inference
        """
        batch_size = len(batched_source_tokens)
        source_instances = [
            self._dataset_reader.get_derivation_step_instance(source_tokens)
            for source_tokens
            in batched_source_tokens
        ]

        batch = Batch(source_instances)
        batch.index_instances(self.vocab)
        input_tensor_dict = batch.as_tensor_dict()
        input_tensor_dict = util.move_to_device(input_tensor_dict, self.device)

        if self._force_decoding_sketch:
            assert all(
                'hyp_program_sketch' in data
                for data in metadata
            )

            force_decoding_sequence = self._dataset_reader.get_program_sketch_tensor_for_force_decoding(
                [
                    data['hyp_program_sketch']
                    for data
                    in metadata
                ],
                batched_source_tokens=batched_source_tokens,
                vocab=self.vocab
            ).to(self.device)
        else:
            force_decoding_sequence = None

        root_level_hypotheses = self.derivation_step_beam_search(
            **input_tensor_dict,
            derivation_level=0,
            force_decoding_sequence=force_decoding_sequence,
            training_derivation_tensor_dicts=training_derivation_tensor_dicts,
            return_attention=return_attention
        )

        if not self._parse_sketch_only:
            all_child_deriv_step_instances = []
            all_child_deriv_parent_decoder_states = {
                'states': [], 'cells': [],
                'attentional_vector': [], 'parent_source_spans': []
            }

            for example_id, root_hyps in enumerate(root_level_hypotheses):
                if self._beam_search_method == 'joint':
                    candidate_root_hyps = root_hyps
                elif self._beam_search_method == 'separate':
                    candidate_root_hyps = [root_hyps[0]]
                else:
                    raise ConfigurationError(self._beam_search_method)

                if self._force_decoding_sketch:
                    candidate_root_hyps = [root_hyps[0]]

                for root_hyp in candidate_root_hyps:
                    child_deriv_steps = root_hyp['child_derivation_steps']
                    # source_tokens = metadata[example_id]['source_tokens']
                    source_tokens = batched_source_tokens[example_id]

                    for slot_name, entry in child_deriv_steps.items():
                        deriv_step_source_tokens = source_tokens[entry['src_span_start_idx']: entry['src_span_end_idx']]
                        # Re-encode the original source utterance if sharing encoding.
                        if self._child_derivation_use_root_utterance_encoding:
                            deriv_step_source_tokens = source_tokens

                        deriv_step_inst = self._dataset_reader.get_derivation_step_instance(
                            deriv_step_source_tokens
                        )
                        all_child_deriv_step_instances.append(deriv_step_inst)
                        all_child_deriv_parent_decoder_states['states'].append(entry['parent_decoder_state'][0])
                        all_child_deriv_parent_decoder_states['cells'].append(entry['parent_decoder_state'][1])
                        all_child_deriv_parent_decoder_states['attentional_vector'].append(entry['parent_decoder_state'][2])
                        all_child_deriv_parent_decoder_states['parent_source_spans'].append([
                            entry['src_span_start_idx'], entry['src_span_end_idx']
                        ])

            if all_child_deriv_step_instances:
                deriv_step_batch = Batch(all_child_deriv_step_instances)
                deriv_step_batch.index_instances(self.vocab)
                deriv_tensor_dict = util.move_to_device(deriv_step_batch.as_tensor_dict(), self.device)

                parent_decoder_state = (
                    torch.stack(all_child_deriv_parent_decoder_states['states'], dim=0),
                    torch.stack(all_child_deriv_parent_decoder_states['cells'], dim=0),
                    torch.stack(all_child_deriv_parent_decoder_states['attentional_vector'], dim=0)
                )
                deriv_tensor_dict['parent_decoder_state'] = parent_decoder_state

                parent_source_spans = all_child_deriv_parent_decoder_states['parent_source_spans']
                deriv_tensor_dict['parent_source_spans'] = parent_source_spans

                n_best = {'separate': 1, 'joint': None}[self._beam_search_method]
                all_child_deriv_step_hypotheses = self.derivation_step_beam_search(
                    **deriv_tensor_dict,
                    n_best=n_best,
                    derivation_level=1,
                    return_attention=return_attention
                )

        continuating_hypotheses = [[] for _ in range(batch_size)]
        cur_deriv_idx = 0
        for example_id, root_hyps in enumerate(root_level_hypotheses):
            if self._beam_search_method == 'joint':
                candidate_root_hyps = root_hyps
            elif self._beam_search_method == 'separate':
                candidate_root_hyps = [root_hyps[0]]
            else:
                raise ConfigurationError(self._beam_search_method)

            if self._force_decoding_sketch:
                candidate_root_hyps = [root_hyps[0]]

            # Drop unnecessary memory-heavy entries since we don't need them anymore
            for hyp in candidate_root_hyps:
                for entry in hyp['child_derivation_steps'].values():
                    del entry['parent_decoder_state']

            for root_hyp_id, root_hyp in enumerate(candidate_root_hyps):
                child_deriv_steps = root_hyp['child_derivation_steps']
                hyp_tokens = list(root_hyp['tokens'])
                hyp_log_prob = root_hyp['log_prob']

                try:
                    root_hyp_representation = self.token_sequence_to_meaning_representation(root_hyp['tokens'])
                except:
                    root_hyp_representation = ['INVALID_TREE']

                derivation: Dict[str, Any] = {
                    'source': batched_source_tokens[example_id],
                    'hyp': root_hyp['tokens'],
                    'log_prob': root_hyp['log_prob'],
                    'beam_position': root_hyp_id,
                    'representation': root_hyp_representation,
                    'parent_source_span_idx': None,
                    'parent_derivation_marker': None,
                    'child_derivation_steps': [],
                    'child_derivation_step_source_span_predictions': root_hyp['child_derivation_steps']
                }

                if return_attention:
                    derivation['attention_weights'] = root_hyp['attention_weights']

                if not self._parse_sketch_only and child_deriv_steps:
                    child_deriv_step_hyps_list: List[Union[Dict, List[Dict]]] = (
                        all_child_deriv_step_hypotheses[
                            cur_deriv_idx:
                            cur_deriv_idx + len(child_deriv_steps)
                        ]
                    )

                    if self._beam_search_method == 'separate':
                        child_deriv_step_hyps_list = [
                            [hyp]
                            for hyp
                            in child_deriv_step_hyps_list
                        ]

                    # enumerate all possible combinations of child derivation hypotheses
                    child_deriv_step_hyps: List[Dict]
                    child_deriv_step_hyps_list: List[List[Dict]]

                    # limit candidates for efficiency concerns
                    child_deriv_step_hyps_list = [hyps_list[:5] for hyps_list in child_deriv_step_hyps_list]
                    for child_deriv_step_hyps in itertools.product(*child_deriv_step_hyps_list):
                        hyp_log_prob = root_hyp['log_prob']
                        hyp_tokens = list(root_hyp['tokens'])
                        child_derivation_steps = []

                        for child_deriv_step_idx, (slot_name, entry) in enumerate(child_deriv_steps.items()):
                            child_deriv_source = entry['src_span_tokens']

                            # pick the best scoring child derivation
                            child_deriv_hyp: Dict = child_deriv_step_hyps[child_deriv_step_idx]   # noqa

                            hyp_tokens = utils.replace_with_sequence(
                                hyp_tokens, [slot_name], child_deriv_hyp['tokens']
                            )

                            hyp_log_prob += child_deriv_hyp['log_prob']

                            child_deriv_step_hyp_entry = {
                                'source': child_deriv_source,
                                'parent_source_span_idx': (entry['src_span_start_idx'], entry['src_span_end_idx']),
                                'parent_derivation_marker': slot_name,
                                'hyp': child_deriv_hyp['tokens'],
                                'log_prob': child_deriv_hyp['log_prob']
                            }

                            if return_attention:
                                child_deriv_step_hyp_entry['attention_weights'] = child_deriv_hyp['attention_weights']

                            child_derivation_steps.append(child_deriv_step_hyp_entry)

                        hyp_derivation = dict(derivation)
                        hyp_derivation['child_derivation_steps'] = child_derivation_steps
                        hyp = {
                            'tokens': hyp_tokens,
                            'log_prob': hyp_log_prob,
                            'derivation': hyp_derivation
                        }
                        continuating_hypotheses[example_id].append(hyp)

                    cur_deriv_idx += len(child_deriv_steps)
                else:
                    hyp = {
                        'tokens': hyp_tokens,
                        'log_prob': hyp_log_prob,
                        'derivation': derivation
                    }

                    continuating_hypotheses[example_id].append(hyp)

        hypotheses = []
        for example_id, hyps in enumerate(continuating_hypotheses):
            # best_hyp = utils.max_by(hyps, key=lambda hyp: hyp['log_prob'])
            # best_hyp['root_level_beam_search_results'] = root_level_hypotheses[example_id]
            # best_hyp['all_hypotheses_in_beam'] = hyps
            ranked_beam = sorted(hyps, key=lambda hyp: hyp['log_prob'], reverse=True)[:self._beam_search.beam_size]
            hypotheses.append(ranked_beam)

        output_dict = {
            "predictions": hypotheses,
        }

        return output_dict

    def derivation_step_beam_search(
        self,
        derivation_step_source_tokens: TextFieldTensors,
        derivation_step_source_spans: torch.LongTensor,
        derivation_step_source_token_first_appearing_indices: torch.LongTensor,
        derivation_step_source_to_target_token_idx_map: torch.LongTensor,
        parent_decoder_state: Tuple[torch.Tensor, torch.Tensor] = None,
        parent_source_spans: List[Tuple] = None,
        metadata: List[Dict[str, Any]] = None,
        n_best: int = None,
        derivation_level: int = 0,
        force_decoding_sequence: torch.LongTensor = None,
        training_derivation_tensor_dicts: List[StateType] = None,
        return_attention: bool = False
    ) -> List[Any]:
        state = self._encode(
            derivation_step_source_tokens, source_spans=derivation_step_source_spans,
            derivation_level=derivation_level
        )
        self._init_decoder_state(state, parent_decoder_state)
        if parent_source_spans:
            state['parent_source_spans'] = parent_source_spans

        total_derivation_step_num, *_ = derivation_step_source_tokens['tokens'][self.source_token_field_key].size()

        # shape: (derivation_steps_num, )
        bos = torch.full(
            (total_derivation_step_num, ), fill_value=self._start_index, dtype=torch.long, device=self.device
        )

        state.update({
            'source_token_first_appearing_indices': derivation_step_source_token_first_appearing_indices,
            'source_to_target_token_idx_map': derivation_step_source_to_target_token_idx_map
        })

        # shape (all_top_k_predictions): (derivation_step_num, beam_size, num_decoding_steps)
        # shape (log_probabilities): (derivation_step_num, beam_size)
        def new_step(
            _last_predictions: torch.Tensor, _state: Dict[str, torch.Tensor], time_step: int
        ):
            return self._beam_search_step(_last_predictions, _state, derivation_level=derivation_level)

        top_k_predictions, log_probabilities, states = self._beam_search._search(
            bos, state,
            new_step,
            return_state=True,
            state_entries_to_log=[
                'decoder_hidden',
                'decoder_cell',
                'decoder_attentional_vec',
                'top_source_span_ids',
                'top_source_span_scores',
            ] + (['attention_weights'] if return_attention else []),
            force_decoding_sequence=force_decoding_sequence
        )

        derivation_step_top_hyp_target_log_prob = log_probabilities.detach().cpu().numpy()
        derivation_step_top_hyps: List[Union[Dict, List[Dict]]] = self._convert_beam_search_results_to_hypotheses(
            top_k_predictions, states,
            derivation_step_source_tokens, derivation_step_source_spans,
            metadata,
            n_best=n_best
        )

        for batch_example_id, (hyp_list, beam_hyp_prob) in enumerate(zip(derivation_step_top_hyps, derivation_step_top_hyp_target_log_prob)):
            if n_best == 1:
                hyp = hyp_list
                hyp['log_prob'] = beam_hyp_prob[0]
            else:
                for hyp_id, hyp in enumerate(hyp_list):
                    hyp['log_prob'] = beam_hyp_prob[hyp_id]

        return derivation_step_top_hyps

    def _beam_search_step(
        self,
        prev_predictions: torch.Tensor,
        state: Dict[str, torch.Tensor],
        derivation_level: int = 0
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
        group_size, source_sequence_length = state["source_token_first_appearing_indices"].size()  # TODO: use another tensor for this

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied,
        # there will be all zeros.

        source_span_idx_offset = self._target_vocab_size + source_sequence_length

        # shape: (group_size,)
        prev_copied_only_mask = (
            (prev_predictions >= self._target_vocab_size) &
            (prev_predictions < source_span_idx_offset)
        )

        # shape: (group_size,)
        prev_source_span_selection_mask = prev_predictions >= source_span_idx_offset

        # If the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        copy_input_choices = torch.full(
            (group_size,), fill_value=self._copy_index, dtype=torch.long, device=self.device
        )

        derivation_marker_input_choices = torch.full(
            (group_size,), fill_value=self._child_derivation_step_marker_idx, dtype=torch.long, device=self.device
        )

        # shape: (group_size,)
        y_tm1 = (
            prev_predictions * ~prev_source_span_selection_mask * ~prev_copied_only_mask +
            derivation_marker_input_choices * prev_source_span_selection_mask +
            copy_input_choices * prev_copied_only_mask
        )

        # shape: (group_size, tgt_embed_size)
        y_tm1_embed = self._get_target_embedder(derivation_level)(y_tm1)

        x_t = torch.cat([y_tm1_embed, state['decoder_attentional_vec']], dim=-1)

        # Update the decoder state by taking a step through the RNN.
        att_vec, state, attention_weights = self._decoder_step(
            x_t, state,
            return_attention=True,
            derivation_level=derivation_level
        )
        state['attention_weights'] = attention_weights

        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(att_vec, derivation_level=derivation_level)

        # Get the un-normalized copy scores for each token in the source sentence,
        # excluding the start and end tokens.
        # shape: (group_size, source_sequence_length)
        copy_scores = self._get_copy_scores(att_vec, state, derivation_level=derivation_level)

        # shape: (group_size, source_sequence_length)
        copy_mask = state["source_tokens_mask"]

        scores_masks_list = [
            # shape: (group_size, target_vocab_size)
            torch.full(generation_scores.size(), True, dtype=torch.bool, device=self.device),
            # shape: (group_size, source_sequence_length)
            copy_mask,
        ]
        scores_list = [
            generation_scores,
            copy_scores
        ]

        compute_source_span_scores = self._compute_source_span_scores_for_derivation_level(derivation_level)

        if compute_source_span_scores:
            scores_masks_list.append(
                # shape: (group_size, source_span_num)
                state['source_spans_mask']
            )

            # shape: (group_size, source_span_num)
            source_span_scores = self._get_new_derivation_step_source_span_probs(att_vec, state, normalize=False)
            scores_list.append(source_span_scores)

        # shape: (group_size, target_vocab_size + source_sequence_length + source_span_num)
        scores_mask = torch.cat(scores_masks_list, dim=-1)

        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_sequence_length + source_span_num)
        all_scores = torch.cat(scores_list, dim=-1)

        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_sequence_length)
        log_probs = util.masked_log_softmax(all_scores, mask=scores_mask)

        if compute_source_span_scores:
            source_spans_num = state['source_spans_mask'].size(1)
            # shape: (group_size, target_vocab_size), (group_size, source_sequence_length), (group_size, source_spans_num)  # noqa
            generation_log_probs, copy_log_probs, source_span_log_probs = log_probs.split(
                [self._target_vocab_size, source_sequence_length, source_spans_num], dim=-1
            )
        else:
            # shape: (group_size, target_vocab_size), (group_size, source_sequence_length)
            generation_log_probs, copy_log_probs = log_probs.split(
                [self._target_vocab_size, source_sequence_length], dim=-1
            )
            source_span_log_probs = None

        # process source span selection probabilities
        # Layout of continuating hypotheses: target_vocab_tokens, source_sequence_length

        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: (group_size, target_vocab_size + source_sequence_length + source_span_num)
        final_log_probs = self._beam_search_step_aggregate_log_probs(
            generation_log_probs, copy_log_probs, source_span_log_probs, state)

        if self._debug and source_span_log_probs is not None:
            # (group_size, source_spans_num)
            top_source_span_scores, top_source_span_ids = torch.topk(
                source_span_log_probs, k=min(source_span_log_probs.size(-1), 5), dim=-1)
            state['top_source_span_scores'] = top_source_span_scores
            state['top_source_span_ids'] = top_source_span_ids

        return final_log_probs, state

    def _beam_search_step_aggregate_log_probs(
        self,
        generation_log_probs: torch.Tensor,
        copy_log_probs: torch.Tensor,
        source_span_log_probs: Optional[torch.Tensor],
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
        if source_span_log_probs is not None:
            # shape: (group_size,)
            generation_log_probs[:, self._child_derivation_step_marker_idx] = float('-inf')
            modified_log_probs_list.append(source_span_log_probs)

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        modified_log_probs = torch.cat(modified_log_probs_list, dim=-1)

        return modified_log_probs

    def _convert_beam_search_results_to_hypotheses(
        self,
        predicted_indices: Union[torch.Tensor, numpy.ndarray],
        batch_states: List[StateType],
        derivation_step_source_tokens: TextFieldTensors,
        derivation_step_source_spans: torch.LongTensor,
        batch_metadata: List[Any],
        n_best: int = None,
    ) -> List[Union[List[Dict], Dict]]:
        """
        Convert predicted indices into tokens.

        predicted_indices: (batch_size, top-K, max_time_step)

        If `n_best = 1`, the result type will be `List[List[str]]`. Otherwise the result
        type will be `List[List[List[str]]]`.
        """
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()

        source_sequence_length = derivation_step_source_tokens['tokens']['mask'].size(1)
        source_span_idx_offset = self._target_vocab_size + source_sequence_length

        derivation_step_source_spans = derivation_step_source_spans.detach().cpu().numpy()

        if predicted_indices.ndim == 2:
            predicted_indices = predicted_indices[:, None, :]

        predictions: List[Union[List[Dict], Dict]] = []
        for example_id, (top_k_predictions, metadata) in enumerate(zip(predicted_indices, batch_metadata)):
            example_predictions: List[Dict] = []
            for k, indices in enumerate(top_k_predictions[:n_best]):
                tokens: List[str] = []
                child_derivation_steps: Dict[str, Any] = collections.OrderedDict()
                indices = list(indices)
                source_tokens = metadata["source_tokens"]
                if self._end_index in indices:
                    indices = indices[: indices.index(self._end_index)]
                for pos, token_index in enumerate(indices):
                    if self._target_vocab_size <= token_index < source_span_idx_offset:
                        adjusted_index = token_index - self._target_vocab_size
                        token = source_tokens[adjusted_index]
                    elif token_index >= source_span_idx_offset:
                        source_span_idx = token_index - source_span_idx_offset
                        source_span_slice_idx = derivation_step_source_spans[example_id, source_span_idx]
                        derivation_step_span_start_idx, derivation_step_span_end_idx = source_span_slice_idx[0], source_span_slice_idx[1]

                        deriv_span_id = len(child_derivation_steps)
                        deriv_marker_name = index_child_derivation_step_marker(deriv_span_id)
                        token = deriv_marker_name

                        if derivation_step_span_end_idx == -1:
                            continue

                        decoder_state_t = (
                            # (decoder_layer_num, hidden_size)
                            batch_states[pos]['decoder_hidden'][example_id][k],
                            batch_states[pos]['decoder_cell'][example_id][k],
                            batch_states[pos]['decoder_attentional_vec'][example_id][k]
                        )

                        child_derivation_steps[deriv_marker_name] = {
                            'src_span_start_idx': derivation_step_span_start_idx,
                            'src_span_end_idx': derivation_step_span_end_idx + 1,  # exclusive indexing
                            'src_span_tokens': source_tokens[derivation_step_span_start_idx: derivation_step_span_end_idx + 1], # noqa
                            'parent_decoder_state': decoder_state_t
                        }

                        if self._debug:
                            # also log the top-K best scored source spans
                            top_source_span_ids = batch_states[pos]['top_source_span_ids'][example_id][k].cpu().tolist()
                            top_source_span_scores = batch_states[pos]['top_source_span_scores'][example_id][k].cpu().tolist()
                            top_hyp_deriv_step_src_spans = []
                            for span_id, span_score in zip(top_source_span_ids, top_source_span_scores):
                                src_span_start, src_span_end = derivation_step_source_spans[example_id, span_id]
                                top_hyp_deriv_step_src_spans.append({
                                    'src_span_start_idx': src_span_start,
                                    'src_span_end_idx': src_span_end,
                                    'src_span_tokens': source_tokens[src_span_start: src_span_end + 1],
                                    'score': span_score
                                })

                            child_derivation_steps[deriv_marker_name]['top_hyp_src_spans'] = top_hyp_deriv_step_src_spans

                    # the following branch could be triggered for models
                    # not using the discrete span selection model
                    elif (
                        not self._compute_span_scores_for_root_derivation
                        and token_index == self._child_derivation_step_marker_idx
                    ):
                        deriv_span_id = len(child_derivation_steps)
                        deriv_marker_name = index_child_derivation_step_marker(deriv_span_id)
                        token = deriv_marker_name

                        decoder_state_t = (
                            # (decoder_layer_num, hidden_size)
                            batch_states[pos]['decoder_hidden'][example_id][k],
                            batch_states[pos]['decoder_cell'][example_id][k],
                            batch_states[pos]['decoder_attentional_vec'][example_id][k]
                        )

                        child_derivation_steps[deriv_marker_name] = {
                            'src_span_start_idx': None,
                            'src_span_end_idx': None,  # exclusive indexing
                            'src_span_tokens': None,
                            'parent_decoder_state': decoder_state_t
                        }
                        # when not using source span modeling, the predicted token is `__SLOT__`
                    else:
                        token = self.vocab.get_token_from_index(token_index, self._target_namespace)

                    tokens.append(token)

                hyp = {
                    'tokens': tokens,
                    'child_derivation_steps': child_derivation_steps,
                }

                if batch_states and 'attention_weights' in batch_states[0]:
                    # shape (target_sequence_len, source_sequence_len)
                    best_prediction_attention_weights = torch.stack(
                        [
                            state['attention_weights'][example_id, k].detach().cpu()
                            for state
                            in batch_states
                        ],
                        dim=0
                    )

                    hyp['attention_weights'] = best_prediction_attention_weights

                example_predictions.append(hyp)

            if n_best == 1:
                predictions.append(example_predictions[0])
            else:
                predictions.append(example_predictions)

        return predictions

    def token_sequence_to_meaning_representation(  # noqa
        self,
        token_sequence: List[str]
    ) -> Sexp:
        mr = seq_to_sexp(token_sequence)

        return mr

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
                if self._dump_full_beam:
                    all_metrics.update(self._oracle_metric.get_metric(reset=reset))
            if self._structured_metric:
                all_metrics.update(self._structured_metric.get_metric(reset=reset))

        return all_metrics


@Predictor.register('decompositional')
class CompositionalParserPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)
