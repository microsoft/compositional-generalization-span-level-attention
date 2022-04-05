import logging
from typing import Dict, Tuple, List, Any, Union, cast

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Metric, BLEU, Perplexity, SequenceAccuracy
from allennlp.nn.beam_search import BeamSearch

from models.beam_search import BeamSearchWithStatesLogging
from models.sequence_metric import SequenceMatchingMetric, SequenceCategorizedMatchMetric

logger = logging.getLogger(__name__)


@Model.register("copynet")
class CopyNet(Model):
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
        beam_size: int,
        max_decoding_steps: int,
        target_embedding_dim: int = 128,
        decoder_hidden_dim: int = 256,
        num_decoder_layers: int = 1,
        decoder_dropout: int = 0.3,
        copy_token: str = "@COPY@",
        source_namespace: str = "source_tokens",
        target_namespace: str = "target_tokens",
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

        # Encoding modules.
        self._source_embedder = source_embedder
        self._encoder = encoder

        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_input_dim = decoder_hidden_dim
        self.decoder_output_dim = self.decoder_hidden_dim = decoder_hidden_dim

        self._target_embedder = Embedding(
            num_embeddings=self._target_vocab_size, embedding_dim=target_embedding_dim
        )

        self._input_projection_layer = Linear(
            target_embedding_dim + self.encoder_output_dim * 2, self.decoder_input_dim
        )

        self._num_decoder_layers = num_decoder_layers
        self._decoder_cell = StackedLSTMCell(self.decoder_input_dim, self.decoder_output_dim, num_decoder_layers, decoder_dropout)

        self._decoder_state_init_linear = Linear(self.encoder_output_dim, self.decoder_output_dim)

        self._attention = attention

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(self.decoder_output_dim, self._target_vocab_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

        # At prediction time, we'll use a beam search to find the best target sequence.
        self._beam_search = BeamSearchWithStatesLogging(
            self._end_index, max_steps=max_decoding_steps, beam_size=beam_size
        )

        self._bleu = BLEU(exclude_indices={self._pad_index, self._start_index, self._end_index})
        # self._sequence_accuracy = SequenceAccuracy()
        self._token_based_metric = SequenceCategorizedMatchMetric()

        initializer(self)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        source_tokens: TextFieldTensors,
        source_token_first_appearing_indices: torch.LongTensor,
        metadata: List[Dict[str, Any]],
        target_tokens: TextFieldTensors = None,
        target_token_first_appearing_indices: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        state = self._encode(source_tokens)

        state.update({
            # 'source_tokens': source_tokens,
            'source_token_first_appearing_indices': source_token_first_appearing_indices
        })

        if target_tokens:
            # state.update({
            #     'target_tokens': target_tokens,
            #     'target_token_first_appearing_indices': target_token_first_appearing_indices
            # })

            state = self._init_decoder_state(state)

            # shape: (batch_size, target_sequence_length)
            target_tokens_mask = util.get_text_field_mask(target_tokens)

            output_dict = self._forward_loss(target_tokens, target_tokens_mask, target_token_first_appearing_indices, state)
        else:
            output_dict = {}

        output_dict["metadata"] = metadata

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self.inference(state)
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
                    hyp_tokens = self._get_predicted_tokens(best_predictions, metadata, n_best=1)
                    ref_tokens = self._get_predicted_tokens(gold_token_ids[:, 1:], metadata, n_best=1)
                    self._token_based_metric(hyp_tokens, ref_tokens, [x['tags'] for x in metadata])

                # # pad sequences to the same length
                # max_target_sequence_length = gold_token_ids.size(-1)
                # if max_decoding_timestep < max_target_sequence_length:
                #     top_k_predictions = torch.cat(
                #         [
                #             top_k_predictions,
                #             top_k_predictions.new_zeros(batch_size, top_k, max_target_sequence_length - max_decoding_timestep)
                #         ],
                #         dim=-1
                #     )
                # elif max_decoding_timestep > max_target_sequence_length:
                #     gold_token_ids = torch.cat([
                #         gold_token_ids,
                #         gold_token_ids.new_zeros(batch_size, max_decoding_timestep - max_target_sequence_length)
                #     ], dim=-1)
                #
                #     target_tokens_mask = torch.cat([
                #         target_tokens_mask,
                #         gold_token_ids.new_full((batch_size, max_decoding_timestep - max_target_sequence_length), fill_value=False, dtype=torch.bool)
                #     ], dim=-1)

        return output_dict

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

        return {"source_mask": source_mask, "encoder_outputs": encoder_outputs}

    def _init_decoder_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, encoder_output_dim)
        encoder_last_states = util.get_final_encoder_states(
            state['encoder_outputs'], state['source_mask'], self._encoder.is_bidirectional())

        decoder_init_state = torch.tanh(self._decoder_state_init_linear(encoder_last_states))
        decoder_init_cell = torch.zeros(state['encoder_outputs'].size(0), self.decoder_output_dim, device=self.device)

        state['attention_weights'] = torch.zeros(state['source_mask'].size(), device=self.device)

        state.update({
            'decoder_hidden': decoder_init_state,
            'decoder_cell': decoder_init_cell
        })

        return state

    def _forward_loss(
        self,
        target_tokens: TextFieldTensors,
        target_tokens_mask: torch.BoolTensor,
        target_token_first_appearing_indices: torch.LongTensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the loss against gold targets.
        """
        batch_size, target_sequence_length = target_tokens["tokens"]["tokens"].size()

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1
        # We use this to fill in the copy index when the previous input was copied.
        # shape: (batch_size,)
        copy_input_choices = cast(torch.LongTensor, torch.full(
            (batch_size,), fill_value=self._copy_index, dtype=torch.long, device=self.device
        ))
        # shape: (batch_size, max_input_sequence_length)
        copy_mask = cast(torch.BoolTensor, source_mask[:, :])
        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        # shape: (batch_size, max_input_sequence_length)
        selective_weights = torch.zeros(copy_mask.size(), device=self.device)

        # Indicates which tokens in the source sentence match the current target token.
        # shape: (batch_size, max_input_sequence_length)
        target_token_source_position_mask = torch.zeros(copy_mask.size(), device=self.device)

        # This is just a tensor of ones which we use repeatedly in `self._get_ll_contrib`,
        # so we create it once here to avoid doing it over-and-over.
        generation_scores_mask = cast(torch.BoolTensor, torch.full(
            (batch_size, self._target_vocab_size), fill_value=1.0, dtype=torch.bool, device=self.device
        ))

        # shape: (batch_size, max_source_sequence_len)
        source_token_first_appearing_indices = torch.zeros()

        # shape: (batch_size, max_target_sequence_len, max_source_sequence_len)
        source_token_first_appearing_indices_expanded = source_token_first_appearing_indices.unsqueeze(1).expand(-1, target_sequence_length, -1)

        # shape: (batch_size, max_target_sequence_len, max_source_sequence_len)
        target_token_source_position_mask = source_token_first_appearing_indices_expanded == target_token_first_appearing_indices.unsqueeze(-1)

        step_log_likelihoods = []
        for timestep in range(num_decoding_steps):
            # shape: (batch_size,)
            input_choices = target_tokens["tokens"]["tokens"][:, timestep]
            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know
            # it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                # shape: (batch_size,)
                copied = (
                    (input_choices == self._oov_index) & (target_token_source_position_mask.sum(-1) > 0)
                ).long()
                # shape: (batch_size,)
                input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                # shape: (batch_size, max_input_sequence_length)
                target_token_source_position_mask = cast(
                    torch.BoolTensor, (
                        state["source_token_first_appearing_indices"] ==
                        target_token_first_appearing_indices[:, timestep + 1].unsqueeze(-1)
                    )
                )
            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)
            # Get generation scores for each token in the target vocab.
            # shape: (batch_size, target_vocab_size)
            generation_scores = self._get_generation_scores(state)
            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            # shape: (batch_size, max_input_sequence_length)
            copy_scores = self._get_copy_scores(state)
            # shape: (batch_size,)
            step_target_tokens = cast(torch.LongTensor, target_tokens["tokens"]["tokens"][:, timestep + 1])
            step_log_likelihood, selective_weights = self._get_ll_contrib(
                generation_scores,
                generation_scores_mask,
                copy_scores,
                step_target_tokens,
                target_token_source_position_mask,
                copy_mask,
            )

            step_log_likelihoods.append(step_log_likelihood.unsqueeze(1))

        # Gather step log-likelihoods.
        # shape: (batch_size, num_decoding_steps = target_sequence_length - 1)
        log_likelihoods = torch.cat(step_log_likelihoods, 1)
        # Get target mask to exclude likelihood contributions from timesteps after
        # the END token.

        # The first timestep is just the START token, which is not included in the likelihoods.
        # shape: (batch_size, num_decoding_steps)
        target_mask = target_tokens_mask[:, 1:]
        # Sum of step log-likelihoods.
        # shape: (batch_size,)
        log_likelihood = (log_likelihoods * target_mask).sum(dim=-1)
        # The loss is the negative log-likelihood, averaged over the batch.
        loss = -log_likelihood.sum() / batch_size

        return {"loss": loss}

    def _decoder_step(
        self,
        prev_predictions: torch.LongTensor,
        selective_weights: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs_mask = state["source_mask"]
        # shape: (batch_size, target_embedding_dim)
        embedded_input = self._target_embedder(prev_predictions)
        # shape: (batch_size, max_input_sequence_length)
        attentive_weights = self._attention(
            state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask
        )
        # shape: (batch_size, encoder_output_dim)
        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
        # shape: (batch_size, encoder_output_dim)
        selective_read = util.weighted_sum(state["encoder_outputs"], selective_weights)
        # shape: (batch_size, target_embedding_dim + encoder_output_dim * 2)
        decoder_input = torch.cat((embedded_input, attentive_read, selective_read), -1)
        # shape: (batch_size, decoder_input_dim)
        projected_decoder_input = self._input_projection_layer(decoder_input)

        state["decoder_hidden"], state["decoder_cell"] = self._decoder_cell(
            projected_decoder_input, (state["decoder_hidden"], state["decoder_cell"])
        )

        attention_weights = attentive_weights
        # attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + util.tiny_value_of_dtype(torch.float))
        # state['attention_weights'] = torch.cat([state['attention_weights'], attention_weights], dim=1)
        state['attention_weights'] = attention_weights

        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._output_generation_layer(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # shape: (batch_size, max_input_sequence_length, decoder_output_dim)
        copy_projection = self._output_copying_layer(state["encoder_outputs"])
        # shape: (batch_size, max_input_sequence_length, decoder_output_dim)
        copy_projection = torch.tanh(copy_projection)
        # shape: (batch_size, max_input_sequence_length)
        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)

        return copy_scores

    def _get_ll_contrib(
        self,
        generation_scores: torch.Tensor,
        generation_scores_mask: torch.BoolTensor,
        copy_scores: torch.Tensor,
        target_token_ids: torch.LongTensor,
        target_token_source_position_mask: torch.BoolTensor,
        copy_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the log-likelihood contribution from a single timestep.

        # Parameters

        generation_scores : `torch.Tensor`
            Shape: `(batch_size, target_vocab_size)`
        generation_scores_mask : `torch.BoolTensor`
            Shape: `(batch_size, target_vocab_size)`. This is just a tensor of 1's.
        copy_scores : `torch.Tensor`
            Shape: `(batch_size, trimmed_source_length)`
        target_token_ids : `torch.LongTensor`
            Shape: `(batch_size,)`
        target_token_source_position_mask : `torch.Tensor`
            Shape: `(batch_size, trimmed_source_length)`
        copy_mask : `torch.BoolTensor`
            Shape: `(batch_size, trimmed_source_length)`

        # Returns

        Tuple[torch.Tensor, torch.Tensor]
            Shape: `(batch_size,), (batch_size, trimmed_source_length)`
        """
        _, target_vocab_size = generation_scores.size()

        # The point of this mask is to just mask out all source token scores
        # that just represent padding. We apply the mask to the concatenation
        # of the generation scores and the copy scores to normalize the scores
        # correctly during the softmax.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat((generation_scores_mask, copy_mask), dim=-1)
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)
        # Globally normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)
        # Calculate the log probability (`copy_log_probs`) for each token in the source sentence
        # that matches the current target token. We use the sum of these copy probabilities
        # for matching tokens in the source sentence to get the total probability
        # for the target token. We also need to normalize the individual copy probabilities
        # to create `selective_weights`, which are used in the next timestep to create
        # a selective read state.
        # shape: (batch_size, trimmed_source_length)
        copy_log_probs = (
            log_probs[:, target_vocab_size:]
            + (
                target_token_source_position_mask.to(log_probs.dtype) + util.tiny_value_of_dtype(log_probs.dtype)
            ).log()
        )
        # Since `log_probs[:, target_vocab_size]` gives us the raw copy log probabilities,
        # we use a non-log softmax to get the normalized non-log copy probabilities.
        # shape: (batch_size, trimmed_source_length)
        selective_weights = util.masked_softmax(log_probs[:, target_vocab_size:], target_token_source_position_mask)
        # This mask ensures that item in the batch has a non-zero generation probabilities
        # for this timestep only when the gold target token is not OOV or there are no
        # matching tokens in the source sentence.
        # shape: (batch_size, 1)
        gen_mask = (target_token_ids != self._oov_index) | (target_token_source_position_mask.sum(-1) == 0)
        log_gen_mask = (gen_mask + util.tiny_value_of_dtype(log_probs.dtype)).log().unsqueeze(-1)
        # Now we get the generation score for the gold target token.
        # shape: (batch_size, 1)
        generation_log_probs = log_probs.gather(1, target_token_ids.unsqueeze(1)) + log_gen_mask
        # ... and add the copy score to get the step log likelihood.
        # shape: (batch_size, 1 + trimmed_source_length)
        combined_gen_and_copy = torch.cat((generation_log_probs, copy_log_probs), dim=-1)
        # shape: (batch_size,)
        step_log_likelihood = util.logsumexp(combined_gen_and_copy)

        return step_log_likelihood, selective_weights

    def inference(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform beam search inference
        """

        batch_size, trimmed_source_length = state['source_mask'].size()

        # shape: (batch_size,)
        bos = torch.full(
            (batch_size,), fill_value=self._start_index, dtype=torch.long, device=self.device
        )

        init_copy_log_probs = (
            torch.zeros((batch_size, trimmed_source_length), device=self.device) + util.tiny_value_of_dtype(torch.float)
        ).log()
        state['copy_log_probs'] = init_copy_log_probs

        # shape (all_top_k_predictions): (batch_size, beam_size, num_decoding_steps)
        # shape (log_probabilities): (batch_size, beam_size)
        all_top_k_predictions, log_probabilities, states = self._beam_search.search(
            bos, state, self._beam_search_step
        )

        # extract attention maps
        # (batch_size, target_sequence_len, source_sequence_len)
        best_prediction_attention_weights = torch.stack(
            [state['attention_weights'][:, 0] for state in states],
            dim=1
        )

        copy_probs = torch.stack(
            [state['copy_log_probs'][:, 0] for state in states],
            dim=1
        ).exp()

        return {
            "predicted_log_probs": log_probabilities, "predictions": all_top_k_predictions,
            'best_prediction_attention_weights': best_prediction_attention_weights,
            'best_prediction_copy_probs': copy_probs
        }

    def _beam_search_step(
        self, prev_predictions: torch.Tensor, state: Dict[str, torch.Tensor]
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
        group_size, trimmed_source_length = state["source_token_first_appearing_indices"].size()

        # Get input to the decoder RNN and the selective weights. `input_choices`
        # is the result of replacing target OOV tokens in `last_predictions` with the
        # copy symbol. `selective_weights` consist of the normalized copy probabilities
        # assigned to the source tokens that were copied. If no tokens were copied,
        # there will be all zeros.

        # shape: (group_size,)
        prev_only_copied_mask = prev_predictions >= self._target_vocab_size

        # If the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (group_size,)
        copy_input_choices = torch.full(
            (group_size,), fill_value=self._copy_index, dtype=torch.long, device=self.device
        )
        # shape: (group_size,)
        input_choices = prev_predictions * ~prev_only_copied_mask + copy_input_choices * prev_only_copied_mask

        # shape: (group_size,)
        adjusted_prev_predictions = prev_predictions - self._target_vocab_size

        # The adjusted indices for items that were not copied will be negative numbers,
        # and therefore invalid. So we zero them out.
        # shape: (group_size,)
        prev_copied_token_positions = adjusted_prev_predictions * prev_only_copied_mask

        # shape: (group_size, trimmed_source_length)
        source_token_first_appearing_indices = state['source_token_first_appearing_indices']
        # shape: (group_size, 1)
        prev_copied_token_first_appearing_positions = source_token_first_appearing_indices.gather(
            dim=1,
            index=prev_copied_token_positions.unsqueeze(-1)
        )

        # shape: (group_size, trimmed_source_length)
        prev_token_copy_position_mask = source_token_first_appearing_indices == prev_copied_token_first_appearing_positions

        # Since we zero'd-out indices for predictions that were not copied,
        # we need to zero out all entries of this mask corresponding to those predictions.
        prev_token_copy_position_mask = prev_token_copy_position_mask & prev_only_copied_mask.unsqueeze(-1)

        prev_selective_weights = util.masked_softmax(state['copy_log_probs'], prev_token_copy_position_mask)

        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, prev_selective_weights, state)

        # Get the un-normalized generation scores for each token in the target vocab.
        # shape: (group_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)

        # Get the un-normalized copy scores for each token in the source sentence,
        # excluding the start and end tokens.
        # shape: (group_size, trimmed_source_length)
        copy_scores = self._get_copy_scores(state)

        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        all_scores = torch.cat((generation_scores, copy_scores), dim=-1)

        # shape: (group_size, trimmed_source_length)
        copy_mask = state["source_mask"]

        # shape: (group_size, target_vocab_size + trimmed_source_length)
        mask = torch.cat(
            (
                # shape: (group_size, target_vocab_size)
                torch.full(generation_scores.size(), True, dtype=torch.bool, device=self.device),
                # shape: (group_size, trimmed_source_sequence_length)
                copy_mask,
            ),
            dim=-1,
        )

        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + trimmed_source_length)
        log_probs = util.masked_log_softmax(all_scores, mask)

        # shape: (group_size, target_vocab_size), (group_size, trimmed_source_length)
        generation_log_probs, copy_log_probs = log_probs.split(
            [self._target_vocab_size, trimmed_source_length], dim=-1
        )

        print('=====First Batch Item, Best Beam Vocab Prob=====')
        vocab_prob_np = generation_log_probs[0].detach().exp().cpu().numpy()
        idx = numpy.argmax(vocab_prob_np)
        print(idx, vocab_prob_np[idx], self.vocab.get_token_from_index(idx, 'target_tokens'))

        copy_probs = copy_log_probs.exp().detach().cpu()[0]
        print('=====First Batch Item, Best Beam Copy Prob=====')
        print(', '.join([str(copy_probs[i].item()) for i in range(copy_probs.size(0))]))

        # Update copy_probs needed for getting the `selective_weights` at the next timestep.
        state["copy_log_probs"] = copy_log_probs

        # We now have normalized generation and copy scores, but to produce the final
        # score for each token in the extended vocab, we have to go through and add
        # the copy scores to the generation scores of matching target tokens, and sum
        # the copy scores of duplicate source tokens.
        # shape: (group_size, target_vocab_size + trimmed_source_length)
        final_log_probs = self._beam_search_step_aggregate_log_probs(generation_log_probs, copy_log_probs, state)

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
            Shape: `(group_size, trimmed_source_length)`
        state : `Dict[str, torch.Tensor]`

        # Returns

        torch.Tensor
            Shape: `(group_size, target_vocab_size + trimmed_source_length)`.
        """
        source_token_first_appearing_indices = state["source_token_first_appearing_indices"]
        group_size, trimmed_source_length = source_token_first_appearing_indices.size()

        # shape: [(group_size, *)]
        modified_log_probs_list: List[torch.Tensor] = []
        for i in range(trimmed_source_length):
            # shape: (group_size,)
            copy_log_probs_slice = copy_log_probs[:, i]

            # We have to combine copy scores for duplicate source tokens so that
            # we can find the overall most likely source token. So, if this is the first
            # occurrence of this particular source token, we add the log_probs from all other
            # occurrences, otherwise we zero it out since it was already accounted for.
            if i < (trimmed_source_length - 1):
                # Sum copy scores from future occurrences of source token.
                # shape: (group_size, trimmed_source_length - i)
                source_future_occurrences_mask = (
                    source_token_first_appearing_indices[:, (i + 1):] ==
                    source_token_first_appearing_indices[:, i].unsqueeze(-1)
                )

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

            modified_log_probs_list.append(copy_log_probs_slice.unsqueeze(-1))

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

        return all_metrics

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Finalize predictions.

        After a beam search, the predicted indices correspond to tokens in the target vocabulary
        OR tokens in source sentence. Here we gather the actual tokens corresponding to
        the indices.
        """
        predicted_tokens = self._get_predicted_tokens(
            output_dict["predictions"], output_dict["metadata"]
        )
        output_dict["predicted_tokens"] = predicted_tokens

        return output_dict
