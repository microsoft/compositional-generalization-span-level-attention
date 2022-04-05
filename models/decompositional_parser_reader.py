import json
from typing import List, Dict, Tuple, Iterable, Optional, Union

import torch
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from dataflow.core.linearize import lispress_to_seq
from overrides import overrides
import logging
from pathlib import Path
import numpy as np

from allennlp.data import Vocabulary, Field
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, ListField, \
    SpanField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token

from dataflow.core.lispress import parse_lispress

from models import utils
from models.derivation_field import DerivationField
from models.seq2seq_parser_reader import SequenceToSequenceModelWithCopyReader, MeaningRepresentationField
from models.utils import find_sequence

logger = logging.getLogger(__name__)


CHILD_DERIVATION_STEP_MARKER = '__SLOT__'


def index_child_derivation_step_marker(idx: int) -> str:
    return f'__SLOT{idx}__'


@DatasetReader.register('decompositional')
class DecompositionalParserReader(SequenceToSequenceModelWithCopyReader):
    def __init__(
        self,
        max_source_span_width: Optional[int] = None,
        pretrained_encoder_name: Optional[str] = None,
        attention_regularization: Optional[str] = None,
        child_derivation_use_root_utterance_encoding: Optional[bool] = False,
        program_sketch_prediction_file: Optional[str] = None,
        **kwargs,
    ):
        super(DecompositionalParserReader, self).__init__(
            pretrained_encoder_name=pretrained_encoder_name,
            attention_regularization=attention_regularization,
            **kwargs
        )

        self._max_source_span_width = max_source_span_width
        self._child_derivation_use_root_utterance_encoding = child_derivation_use_root_utterance_encoding
        self._program_sketch_prediction_file = program_sketch_prediction_file

        # initialize empty fields in DerivationField
        empty_src_text_field = TextField([], self._source_token_indexers).empty_field()
        DerivationField.empty_fields = {
            'derivation_step_source_tokens': ListField([empty_src_text_field]).empty_field(),
            # source spans are represented as a nested `ListField[ListField[SpanField]]`
            # for each derivation step, for each span
            'derivation_step_source_spans': ListField([
                ListField([
                    SpanField(-1, -1, empty_src_text_field)
                ]).empty_field()
            ]).empty_field(),
            'derivation_step_source_to_target_token_idx_map': ListField([
                NamespaceSwappingField([], target_namespace='target_tokens').empty_field()
            ]).empty_field(),
            'derivation_step_source_token_first_appearing_indices': ListField([
                ArrayField(np.array([]), dtype=np.float32).empty_field()
            ]).empty_field(),
            'derivation_step_target_tokens': ListField([
                TextField([], {'tokens': self._target_token_indexer}).empty_field()
            ]).empty_field(),
            'derivation_step_target_token_first_appearing_indices': ListField([
                ArrayField(np.array([]), dtype=np.float32).empty_field()
            ]).empty_field(),
            'derivation_step_target_token_source_span_id': ListField([
                ArrayField(np.array([]), padding_value=-1, dtype=np.long).empty_field()
            ]).empty_field(),
            'derivation_step_target_to_source_alignment': ListField([
                ArrayField(np.array([]), padding_value=-1).empty_field()
            ]).empty_field()
        }

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)

        if self._program_sketch_prediction_file:
            sketch_pred_results = utils.load_jsonl_file(self._program_sketch_prediction_file)

        with file_path.open('r') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")
                if not line:
                    continue

                if line.startswith('{'):
                    data = json.loads(line)
                    source_sequence = data['source']
                    target_sequence = data['target']
                    tags = data['tags']
                    decomposition = data['decomposition']
                    token_level_alignment_info = data['token_level_alignments']
                else:
                    line_parts = line.split("\t")

                    source_sequence, tags_str, target_sequence = line_parts[1], line_parts[2], line_parts[3]
                    tags = list(set(tags_str[1:-1].split(', ')))
                    decomposition = token_level_alignment_info = None

                metafields = dict()
                if self._program_sketch_prediction_file:
                    metafields['hyp_program_sketch'] = sketch_pred_results[line_num]['hyp_program_sketches'][0]

                yield self.text_to_instance(
                    source_sequence, tags, target_sequence,
                    derivation_info=decomposition,
                    example_idx=line_num,
                    metafields=metafields,
                    token_level_alignment_info=token_level_alignment_info
                )

    @overrides
    def text_to_instance(
        self,
        source_sentence: str,
        tags: List[str] = None,
        target_sentence: str = None,
        derivation_info: Dict = None,
        token_level_alignment_info: List[Dict] = None,
        example_idx: str = None,
        metafields: Dict = None
    ) -> Instance:
        tags = tags or list()

        source_tokens_on_white_space: List[Token] = self._white_space_tokenizer.tokenize(source_sentence)
        if self.use_pretrained_encoder:
            source_tokens, source_subtoken_offsets = self._source_tokenizer.intra_word_tokenize(
                [tok.text for tok in source_tokens_on_white_space]
            )
        else:
            source_tokens = source_tokens_on_white_space
            source_subtoken_offsets = None

        source_field = TextField(source_tokens, self._source_token_indexers)

        fields_dict = {
            "source_tokens": source_field,
        }

        meta_dict = {
            'source_tokens': [token.text for token in source_tokens],
            'source_tokens_on_white_space': [token.text for token in source_tokens_on_white_space],
            'source_subtoken_offsets': source_subtoken_offsets,
            'tags': tags
        }

        meta_dict.update(metafields or dict())

        if example_idx is not None:
            meta_dict['example_idx'] = example_idx

        if target_sentence is not None:
            target_tokens, target_metadata = self._tokenize_target(target_sentence)
            target_tokens_on_whitespace = target_sentence.split(' ')
            meta_dict['target_tokens'] = [tok.text for tok in target_tokens]
            meta_dict['target_representation'] = parse_lispress(target_sentence)

        if derivation_info is not None:
            # process source/target decomposition
            root_deriv_source_tokens = source_tokens
            child_derivation_field_dict_list = []
            child_deriv_steps = []
            num_derivation_levels = 2
            # a list of two derivation levels
            derivation = [[], []]

            root_deriv_target_tokens, root_deriv_target_metadata = self._tokenize_target(
                derivation_info['sketch_sexp'])
            root_deriv_target_tokens_on_whitespace = derivation_info['sketch_sexp'].split(' ')

            for slot_name, child_deriv_step in derivation_info['named_sub_sexp'].items():
                deriv_step_target_tokens, deriv_step_target_metadata = (
                    self._tokenize_target(child_deriv_step['sub_sexp'])
                )
                deriv_step_target_tokens_on_whitespace = child_deriv_step['sub_sexp'].split(' ')
                target_span_position = [
                    idx
                    for idx, x in enumerate(root_deriv_target_tokens)
                    if x.text == slot_name
                ][0]

                is_floating_child_derivation = child_deriv_step['span_alignment'] is None

                deriv_step_source_tokens = None
                if not is_floating_child_derivation:
                    source_span_position = tuple(child_deriv_step['span_alignment']['source_span'])
                    src_span_subtokens, (src_subtoken_start, src_subtoken_end) = self._get_subtokens_slice(
                        source_tokens, (source_span_position[0], source_span_position[1]), source_subtoken_offsets)
                    source_span_subtoken_position = (src_subtoken_start, src_subtoken_end)
                    deriv_step_source_tokens = source_tokens[src_subtoken_start: src_subtoken_end]
                else:
                    deriv_step_source_tokens = source_tokens
                    source_span_position = source_span_subtoken_position = None

                if self._child_derivation_use_root_utterance_encoding:
                    deriv_step_source_tokens = source_tokens

                assert deriv_step_source_tokens is not None

                child_derivation_field_dict = self._get_derivation_fields_dict(
                    deriv_step_source_tokens,
                    deriv_step_target_tokens,
                    target_subtoken_offsets=deriv_step_target_metadata['subtoken_offsets'],
                    target_source_span_mentions=deriv_step_target_metadata['source_span_mentions'],
                )

                if (
                    self._attention_regularization and
                    self._child_derivation_use_root_utterance_encoding
                ):
                    align_matrix = np.zeros((len(deriv_step_target_tokens) + 2, len(source_tokens)))  # account for <s> and </s>
                    align_mat_view = align_matrix[1:-1]
                    alignment_padding_val = -1
                    if not is_floating_child_derivation:
                        use_segment_level_regularization = ':token:' not in self._attention_regularization
                        if use_segment_level_regularization:
                            align_matrix[1:-1, src_subtoken_start: src_subtoken_end] = 1.0  # noqa
                        else:
                            assert target_tokens_on_whitespace  # noqa
                            child_deriv_index_in_original_target = child_deriv_step['sub_tree_start_idx_in_original_target']
                            for entry in token_level_alignment_info:
                                if source_span_position[0] <= entry['source_token_idx'] < source_span_position[1]:
                                    _src_sub_tokens, (_src_subtoken_start, _src_subtoken_end) = self._get_subtokens_slice(
                                        source_tokens,
                                        (entry['source_token_idx'], entry['source_token_idx'] + 1),
                                        source_subtoken_offsets
                                    )

                                    tgt_token_idx = entry['target_token_idx']
                                    if child_deriv_index_in_original_target[0] <= tgt_token_idx < child_deriv_index_in_original_target[1]:
                                        tgt_token_relative_idx = tgt_token_idx - child_deriv_index_in_original_target[0]
                                        assert deriv_step_target_tokens_on_whitespace[tgt_token_relative_idx] == entry['target_token']

                                        _tgt_sub_tokens, (_tgt_subtoken_start, _tgt_subtoken_end) = self._get_subtokens_slice(
                                            deriv_step_target_tokens,
                                            (tgt_token_relative_idx, tgt_token_relative_idx + 1),
                                            deriv_step_target_metadata['subtoken_offsets']
                                        )

                                        align_mat_view[
                                            _tgt_subtoken_start: _tgt_subtoken_end,
                                            _src_subtoken_start: _src_subtoken_end
                                        ] = 1.0

                        for i in range(align_matrix.shape[0]):
                            if sum(align_matrix[i]) == 0:
                                align_matrix[i] = alignment_padding_val
                    else:
                        align_matrix.fill(alignment_padding_val)

                    alignment_field = ArrayField(align_matrix, padding_value=-1)
                    child_derivation_field_dict['derivation_step_target_to_source_alignment'] = alignment_field

                child_derivation_field_dict_list.append(child_derivation_field_dict)

                child_deriv_steps.append({
                    'source_tokens': [tok.text for tok in deriv_step_source_tokens],
                    'target_tokens': [tok.text for tok in deriv_step_target_tokens],
                    'is_floating': is_floating_child_derivation,
                    'target_metadata': deriv_step_target_metadata,
                    'source_span_position': source_span_subtoken_position,
                    'source_space_tokenized_span_position': source_span_position,
                    'target_span_position': target_span_position,
                    'slot_name': slot_name
                })

            root_field_dict = self._get_derivation_fields_dict(
                root_deriv_source_tokens, root_deriv_target_tokens,
                child_derivation_steps=child_deriv_steps,
                target_subtoken_offsets=root_deriv_target_metadata['subtoken_offsets'],
                target_source_span_mentions=root_deriv_target_metadata['source_span_mentions']
            )

            # process source-target alignment information for attention regularization
            if self._attention_regularization:
                alignment_padding_val = -1
                align_matrix = np.zeros((len(root_deriv_target_tokens) + 2, len(source_tokens)))
                align_mat_view = align_matrix[1:-1]
                alignment_spans = []

                use_token_level_regularization = ':token:' in self._attention_regularization
                use_segment_level_regularization = not use_token_level_regularization

                if use_segment_level_regularization:
                    for entry in derivation_info['alignments']:
                        tgt_token_start, tgt_token_end = entry['target_tokens_idx']
                        src_token_start, src_token_end = entry['source_tokens_idx']

                        tgt_span_subtokens, (tgt_subtoken_start, tgt_subtoken_end) = self._get_subtokens_slice(
                            root_deriv_target_tokens,  # account for <s> and </s>
                            (tgt_token_start, tgt_token_end), root_deriv_target_metadata['subtoken_offsets']
                        )

                        src_span_subtokens, (src_subtoken_start, src_subtoken_end) = self._get_subtokens_slice(
                            source_tokens,
                            (src_token_start, src_token_end), source_subtoken_offsets
                        )

                        alignment_spans.append({
                            'tgt_span': (tgt_subtoken_start, tgt_subtoken_end),
                            'src_span': (src_subtoken_start, src_subtoken_end)
                        })

                        if ':inner:' in self._attention_regularization or ':all:' in self._attention_regularization:
                            # wrapped by <s> and </s>
                            align_mat_view[tgt_subtoken_start: tgt_subtoken_end, src_subtoken_start: src_subtoken_end] = 1.0

                    if ':outer:' in self._attention_regularization or ':all:' in self._attention_regularization:
                        source_token_alignment_mask_for_sketch_token = np.ones(len(source_tokens))

                        for alignment in alignment_spans:
                            source_token_alignment_mask_for_sketch_token[
                                alignment['src_span'][0]: alignment['src_span'][1]
                            ] = 0

                        for tgt_token_idx in range(align_mat_view.shape[0]):
                            is_sketch_token = not(
                                any(
                                    alignment['tgt_span'][0] <= tgt_token_idx < alignment['tgt_span'][1]
                                    for alignment
                                    in alignment_spans
                                )
                            )

                            if is_sketch_token:
                                align_mat_view[tgt_token_idx] = source_token_alignment_mask_for_sketch_token
                elif use_token_level_regularization:
                    # assert use_token_level_regularization
                    for entry in token_level_alignment_info:
                        token_belong_to_child_derivation = any(
                            child_deriv_step['sub_tree_start_idx_in_original_target'][0] <= entry['target_token_idx'] < child_deriv_step['sub_tree_start_idx_in_original_target'][1]
                            for child_deriv_step
                            in derivation_info['named_sub_sexp'].values()
                        )

                        if not token_belong_to_child_derivation:
                            source_span, (src_subtoken_start, src_subtoken_end) = self._get_subtokens_slice(
                                source_tokens,
                                (entry['source_token_idx'], entry['source_token_idx'] + 1), source_subtoken_offsets
                            )

                            tgt_token_idx = entry['target_token_idx']
                            tgt_token_relative_idx = tgt_token_idx
                            for child_deriv_step in derivation_info['named_sub_sexp'].values():
                                child_deriv_span_start, child_deriv_span_end = child_deriv_step['sub_tree_start_idx_in_original_target']
                                if child_deriv_span_end <= tgt_token_idx:
                                    tgt_token_relative_idx = tgt_token_relative_idx - (child_deriv_span_end - child_deriv_span_start) + 1

                            try:
                                assert root_deriv_target_tokens_on_whitespace[tgt_token_relative_idx] == entry['target_token']
                            except:
                                logger.warning(f'cannot locate index of target token {entry["target_token"]}')
                                continue

                            target_span, (tgt_subtoken_start, tgt_subtoken_end) = self._get_subtokens_slice(
                                root_deriv_target_tokens,  # account for <s> and </s>
                                (tgt_token_relative_idx, tgt_token_relative_idx + 1), root_deriv_target_metadata['subtoken_offsets']
                            )

                            align_mat_view[tgt_subtoken_start: tgt_subtoken_end, src_subtoken_start: src_subtoken_end] = 1.0

                for i in range(align_mat_view.shape[0]):
                    if sum(align_mat_view[i]) == 0:
                        align_mat_view[i] = alignment_padding_val

                align_matrix[0] = align_matrix[-1] = alignment_padding_val

                alignment_field = ArrayField(align_matrix, padding_value=-1)
                # fields_dict['root_derivation_target_to_source_alignment'] = alignment_field
                root_field_dict['derivation_step_target_to_source_alignment'] = alignment_field
                meta_dict['alignment_info'] = derivation_info['alignments']

            derivation_log = {
                'source': [tok.text for tok in root_deriv_source_tokens],
                'target': [tok.text for tok in root_deriv_target_tokens],
                'child_derivation_steps': child_deriv_steps
            }

            derivation_field_dict_list = [
                [root_field_dict]
            ]
            if child_derivation_field_dict_list:
                derivation_field_dict_list.append(child_derivation_field_dict_list)

            aggregated_derivation_level_fields_dict_list = []
            for level, deriv_level_steps_field_dicts in enumerate(derivation_field_dict_list):
                # aggregate fields to `ListField`s by their keys
                keys = list(deriv_level_steps_field_dicts[0].keys())
                cur_level_field_dict: Dict[str, ListField] = dict()
                for key in keys:
                    field = ListField([
                        d[key]
                        for d
                        in deriv_level_steps_field_dicts
                    ])
                    cur_level_field_dict[key] = field

                aggregated_derivation_level_fields_dict_list.append(cur_level_field_dict)

            derivation_field = DerivationField(aggregated_derivation_level_fields_dict_list)
            fields_dict['derivation'] = derivation_field

            meta_dict.update({
                'derivation': derivation_log
            })

        fields_dict['metadata'] = MetadataField(meta_dict)

        instance = Instance(fields_dict)

        return instance

    def _get_derivation_fields_dict(
        self,
        source_tokens: Union[List[Token], List[str]],
        target_tokens: List[Token] = None,
        *,
        target_subtoken_offsets: List[Tuple[int, int]] = None,
        target_source_span_mentions: List[Tuple[int, int]] = None,
        child_derivation_steps: List[Dict] = None,
        metadata_dict: Dict = None
    ) -> Dict[str, Field]:
        if isinstance(source_tokens[0], str):
            # source tokens have already been sub-tokenized
            source_tokens = [Token(tok) for tok in source_tokens]

            if self.use_pretrained_encoder:
                token_ids = self._source_tokenizer.tokenizer.convert_tokens_to_ids([
                    tok.text for tok in source_tokens])

                for token, token_id in zip(source_tokens, token_ids):
                    token.text_id = token_id

        # enumerate candidate source spans
        source_tokens_field = TextField(source_tokens, self._source_token_indexers)
        source_spans = []
        for span_start, span_end in enumerate_spans(source_tokens, max_span_width=self._max_source_span_width):
            source_spans.append(SpanField(span_start, span_end, source_tokens_field))

        if metadata_dict is not None:
            metadata_dict['source_spans_num'] = len(source_spans)

        fields_dict = {  # noqa
            'derivation_step_source_tokens': source_tokens_field,
            'derivation_step_source_spans': ListField(source_spans),
            'derivation_step_source_to_target_token_idx_map': NamespaceSwappingField(
                source_tokens, target_namespace='target_tokens'
            ),
            'derivation_step_source_token_first_appearing_indices': ArrayField(
                np.array(self._tokens_to_first_appearing_indices(source_tokens))
            )
        }

        if target_tokens:
            child_derivation_steps = child_derivation_steps or []
            target_tokens = [Token(START_SYMBOL)] + target_tokens + [Token(END_SYMBOL)]
            target_deriv_step_source_span_slice_id_array = np.full((len(target_tokens), ), fill_value=-1, dtype=np.long)
            target_tokens = list(target_tokens)

            for child_deriv_step in child_derivation_steps:
                slot_name = child_deriv_step['slot_name']
                deriv_tgt_span_position = child_deriv_step['target_span_position']
                # account for the prepended `START_SYMBOL`
                target_tokens[deriv_tgt_span_position + 1] = Token(CHILD_DERIVATION_STEP_MARKER)

                if not child_deriv_step['is_floating']:
                    deriv_src_span_position = child_deriv_step['source_span_position']
                    # convert exclusive index to inclusive index
                    src_span_start, src_span_end = deriv_src_span_position[0], deriv_src_span_position[1] - 1
                    source_span_slice_idx = source_spans.index((src_span_start, src_span_end))
                    # account for the prepended `START_SYMBOL`
                    target_deriv_step_source_span_slice_id_array[deriv_tgt_span_position + 1] = source_span_slice_idx
                    child_deriv_step['source_span_slice_idx'] = source_span_slice_idx

            target_span_field = MeaningRepresentationField(
                target_tokens,
                token_indexers={'tokens': self._target_token_indexer},
                source_span_mentions=target_source_span_mentions
            )

            source_and_target_token_first_appearing_indices = self._tokens_to_first_appearing_indices(
                source_tokens + target_tokens
            )

            fields_dict.update({
                'derivation_step_target_tokens': target_span_field,
                'derivation_step_source_token_first_appearing_indices': ArrayField(
                    np.array(source_and_target_token_first_appearing_indices[: len(source_tokens)])
                ),
                'derivation_step_target_token_first_appearing_indices': ArrayField(
                    np.array(source_and_target_token_first_appearing_indices[len(source_tokens):])
                ),
                'derivation_step_target_token_source_span_id': ArrayField(
                    target_deriv_step_source_span_slice_id_array, padding_value=-1,
                    dtype=target_deriv_step_source_span_slice_id_array.dtype
                )
            })

        return fields_dict

    def get_derivation_step_instance(self, source_tokens: List[str]) -> Instance:
        fields_dict = self._get_derivation_fields_dict(source_tokens)
        # fields_dict = {
        #     key.partition('decomposed_')[-1]: val
        #     for key, val in fields_dict.items()
        # }

        metadata = MetadataField({
            'source_tokens': source_tokens
        })
        fields_dict['metadata'] = metadata

        instance = Instance(fields_dict)

        return instance

    def get_program_sketch_tensor_for_force_decoding(
        self,
        batched_program_sketches: List[List[str]],
        batched_source_tokens: List[List[str]],
        vocab: Vocabulary,
    ) -> torch.LongTensor:
        batch_size = len(batched_program_sketches)
        max_target_seq_len = max(len(sketch) for sketch in batched_program_sketches) + 2

        sketch_token_array = np.zeros((batch_size, max_target_seq_len), dtype=np.int64)
        # fill the array with id of <eos> since beam search requires it.
        sketch_token_array.fill(vocab.get_token_index(END_SYMBOL, 'target_tokens'))
        copy_tokens_offset = vocab.get_vocab_size('target_tokens')

        for example_id, (sketch, source) in enumerate(
            zip(
                batched_program_sketches,
                batched_source_tokens
            )
        ):
            sketch_token_ids = []
            sketch_token_ids.append(vocab.get_token_index(START_SYMBOL, 'target_tokens'))

            for i in range(len(sketch)):
                tgt_token = sketch[i]

                if tgt_token.startswith('__SLOT'):
                    token_idx = vocab.get_token_index(CHILD_DERIVATION_STEP_MARKER, 'target_tokens')
                elif tgt_token in source:
                    source_token_idx = source.index(tgt_token)
                    token_idx = copy_tokens_offset + source_token_idx
                else:
                    token_idx = vocab.get_token_index(tgt_token, 'target_tokens')

                sketch_token_ids.append(token_idx)

            sketch_token_ids.append(vocab.get_token_index(END_SYMBOL, 'target_tokens'))
            sketch_token_array[example_id, :len(sketch_token_ids)] = sketch_token_ids

        return torch.from_numpy(sketch_token_array)


def main():
    reader = DecompositionalParserReader()
    dataset = reader.read('data/calflow_0608/calflow.orgchart/source_domain_with_target_num32/valid.sketch.heuristic.mixed_granularity.jsonl')
    vocab = Vocabulary.from_instances(dataset.instances[1000:1032])
    dataset.index_with(vocab)

    for inst in dataset:
        inst.index_fields(vocab)

    from allennlp.data import Batch

    batch = Batch(dataset.instances[:10])
    tensor_dict = batch.as_tensor_dict(batch.get_padding_lengths())
    print(tensor_dict)


if __name__ == '__main__':
    main()
