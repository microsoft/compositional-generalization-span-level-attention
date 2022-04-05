import json
from typing import List, Dict, Tuple, Iterable, Set, Optional, Union

from dataflow.core.linearize import seq_to_sexp
from dataflow.core.sexp import Sexp
from overrides import overrides
import logging
import re
from pathlib import Path
from re import Match
import numpy as np

from allennlp.data import Vocabulary
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, SequenceField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer


# from dataflow.core.lispress import parse_lispress

from models.utils import flatten, find_sequence, sexp_to_tokenized_str


logger = logging.getLogger(__name__)


DATASET_NAMES = {'calflow', 'cfq', 'text2sql'}


def ensure_token_list(token_list):
    tokens = []
    for token in token_list:
        if not isinstance(token, list):
            tokens.append(token)
        else:
            tokens.extend(ensure_token_list(token))

    return tokens


class MeaningRepresentationField(TextField):
    def __init__(
        self,
        tokens: List[Token],
        token_indexers: Dict[str, TokenIndexer],
        source_span_mentions: List[Tuple[int, int]] = None,
    ):
        super().__init__(tokens, token_indexers)

        self.source_span_mentions = source_span_mentions or {}

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for token_idx, token in enumerate(self.tokens):
            if self.tokens[0].text == START_SYMBOL:
                token_idx = token_idx - 1

            if any(span[0] <= token_idx < span[1] for span in self.source_span_mentions):
                continue
            else:
                for indexer in self._token_indexers.values():
                    indexer.count_vocab_items(token, counter)


@DatasetReader.register('seq2seq_with_copy')
class SequenceToSequenceModelWithCopyReader(DatasetReader):
    def __init__(
        self,
        pretrained_encoder_name: Optional[str] = None,
        attention_regularization: Optional[str] = None,
        dataset_name: Optional[str] = 'calflow',
        cfq_add_variable_names_to_vocab: Optional[bool] = False,
        add_variable_names_to_vocab: Optional[bool] = False,
        num_examples: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert dataset_name in DATASET_NAMES, f'Invalid dataset {dataset_name}'

        self._white_space_tokenizer = SpacyTokenizer(split_on_spaces=True)
        self._pretrained_encoder_name = pretrained_encoder_name
        self._num_examples = num_examples

        if pretrained_encoder_name:
            self._source_tokenizer = PretrainedTransformerTokenizer(
                pretrained_encoder_name,
                #tokenizer_kwargs={'use_fast': False}
            )
            self._source_token_indexers = {
                'tokens': PretrainedTransformerIndexer(
                    pretrained_encoder_name, namespace='source_tokens'),
            }
        else:
            self._source_tokenizer = SpacyTokenizer(split_on_spaces=True)
            self._source_token_indexers = {
                "tokens": SingleIdTokenIndexer(namespace='source_tokens', lowercase_tokens=True)
            }

        self._cfq_add_variable_names_to_vocab = cfq_add_variable_names_to_vocab
        if cfq_add_variable_names_to_vocab:
            logger.warning('cfq_add_variable_names_to_vocab has been deprecated.')

        self._add_variable_names_to_vocab = (add_variable_names_to_vocab or cfq_add_variable_names_to_vocab)

        if dataset_name == 'calflow':
            self._target_tokenizer = self._source_tokenizer
        elif dataset_name == 'cfq':
            self._target_tokenizer = SpacyTokenizer(split_on_spaces=True)

            if self._add_variable_names_to_vocab:
                special_tokens = [f'm{idx}' for idx in range(0, 100)]
                self._source_tokenizer.tokenizer.add_tokens(special_tokens)
        elif dataset_name == 'text2sql':
            self._target_tokenizer = SpacyTokenizer(split_on_spaces=True)

            if self._add_variable_names_to_vocab:
                from utils.sql.sql_utils.text2sql_utils import CANONICAL_VARIABLES
                variables = sorted(CANONICAL_VARIABLES)
                self._source_tokenizer.tokenizer.add_tokens(variables)
                self._canonical_variable_index = {var: idx for idx, var in enumerate(variables)}

        if 'tokens' not in self._source_token_indexers:
            raise ConfigurationError(
                f"{self} expects 'source_token_indexers' to contain "
                "a token indexer called 'tokens'."
            )

        self._target_token_indexer = SingleIdTokenIndexer(namespace='target_tokens')
        self._attention_regularization = attention_regularization
        self.dataset_name = dataset_name

    @property
    def use_pretrained_encoder(self):
        return self._pretrained_encoder_name is not None

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)

        with file_path.open('r') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")
                if not line:
                    continue

                kwargs = {'use_alignment_information': True}
                if line.startswith('{"') or line.startswith('['):
                    data = json.loads(line)
                    use_source_and_target_derivations = (
                        isinstance(data, list) and 'child_derivation_steps' in data[0]
                        or
                        isinstance(data, dict) and 'child_derivation_steps' in data
                    )
                    use_alignment_info = not use_source_and_target_derivations

                    if use_source_and_target_derivations:
                        data = data[0] if isinstance(data, list) else data

                        source_sequence = data['source']
                        target_sequence = data['target']
                        tags = data['tags']
                        alignment_info = data['child_derivation_steps']
                        kwargs['use_alignment_information'] = False
                    else:
                        source_sequence = data['source']
                        target_sequence = data['target']
                        tags = data['tags']
                        alignment_info = data['alignments']

                    token_level_alignment_info = data.get('token_level_alignments')
                else:
                    line_parts = line.split("\t")

                    source_sequence, tags_str, target_sequence = line_parts[1], line_parts[2], line_parts[3]
                    tags = list(set(tags_str[1:-1].split(', ')))
                    alignment_info = None
                    token_level_alignment_info = None

                yield self.text_to_instance(
                    source_sequence, tags, target_sequence=target_sequence,
                    alignment_info=alignment_info,
                    token_level_alignment_info=token_level_alignment_info,
                    **kwargs
                )

                if self._num_examples and line_num >= self._num_examples:
                    break

    @overrides
    def text_to_instance(
        self,
        source_sentence: str,
        tags: List[str] = None,
        target_sequence: str = None,
        alignment_info: List[Dict] = None,
        use_alignment_information: bool = True,
        token_level_alignment_info: List[Dict] = None,
    ) -> Instance:
        tags = tags or list()

        source_tokens_on_white_space: List[Token] = self._white_space_tokenizer.tokenize(source_sentence)
        if self.use_pretrained_encoder:
            source_tokens, source_subtoken_offsets = self._tokenize_source(source_sentence)
        else:
            source_tokens = source_tokens_on_white_space
            source_subtoken_offsets = None

        source_field = TextField(source_tokens, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_token_idx_map_field = NamespaceSwappingField(
            source_tokens, target_namespace='target_tokens'
        )

        fields_dict = {
            "source_tokens": source_field,
            'source_to_target_token_idx_map': source_to_target_token_idx_map_field,
            "source_token_first_appearing_indices": ArrayField(
                np.array(self._tokens_to_first_appearing_indices(source_tokens))
            )
        }

        meta_dict = {
            'source_tokens': [token.text for token in source_tokens],
            'source_tokens_on_white_space': [token.text for token in source_tokens_on_white_space],
            'source_subtoken_offsets': source_subtoken_offsets,
            'tags': tags
        }

        if target_sequence is not None:
            target_tokens, target_metadata = self._tokenize_target(target_sequence)
            target_tokens_on_whitespace = target_sequence.split(' ')
            # self._test_tokenized_target(target_sequence, target_tokens, target_metadata['subtoken_offsets'])
            target_tokens.insert(0, Token(START_SYMBOL))
            target_tokens.append(Token(END_SYMBOL))

            target_field = MeaningRepresentationField(
                target_tokens, {'tokens': self._target_token_indexer},
                source_span_mentions=target_metadata['source_span_mentions']
            )
            fields_dict["target_tokens"] = target_field

            source_and_target_token_first_appearing_indices = self._tokens_to_first_appearing_indices(
                source_tokens + target_tokens
            )

            source_token_ids = source_and_target_token_first_appearing_indices[: len(source_tokens)]
            fields_dict["source_token_first_appearing_indices"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_first_appearing_indices[len(source_tokens):]
            fields_dict["target_token_first_appearing_indices"] = ArrayField(np.array(target_token_ids))

            meta_dict['target_tokens'] = [token.text for token in target_tokens[1:-1]]

            if self._attention_regularization:
                target_subtoken_offsets = target_metadata['subtoken_offsets']

                use_token_level_regularization = ':token:' in self._attention_regularization
                use_segment_level_regularization = not use_token_level_regularization

                alignment_padding_val = -1
                align_matrix = np.zeros((len(target_tokens), len(source_tokens)))
                align_mat_view = align_matrix[1:-1]
                alignment_spans = []

                if use_segment_level_regularization and alignment_info:
                    for entry in alignment_info:
                        if use_alignment_information:
                            tgt_token_start, tgt_token_end = entry['target_tokens_idx']
                            src_token_start, src_token_end = entry['source_tokens_idx']

                            src_span_subtokens, (src_subtoken_start, src_subtoken_end) = self._get_subtokens_slice(
                                source_tokens,
                                (src_token_start, src_token_end), source_subtoken_offsets
                            )
                        else:
                            if entry.get('is_floating', False):
                                continue

                            src_subtoken_start, src_subtoken_end = entry['source_span_position']

                            if entry['parent_arg_name']:
                                name_arg_val_sexp = [entry['parent_arg_name'], entry['target_sexp']]
                                target_span_sexp_tokens = sexp_to_tokenized_str(name_arg_val_sexp)

                                try:
                                    tgt_token_start, tgt_token_end = find_sequence(
                                        target_tokens_on_whitespace, target_span_sexp_tokens)
                                except IndexError:
                                    # remove last parenthesis
                                    tgt_token_start, tgt_token_end = find_sequence(
                                        target_tokens_on_whitespace, target_span_sexp_tokens[1:-1])
                            else:
                                target_span_sexp_tokens = sexp_to_tokenized_str(entry['target_sexp'])
                                tgt_token_start, tgt_token_end = find_sequence(
                                    target_tokens_on_whitespace, target_span_sexp_tokens)

                        tgt_span_subtokens, (tgt_subtoken_start, tgt_subtoken_end) = self._get_subtokens_slice(
                            target_tokens[1:-1],  # account for <s> and </s>
                            (tgt_token_start, tgt_token_end), target_subtoken_offsets
                        )

                        alignment_spans.append({
                            'tgt_span': (tgt_subtoken_start, tgt_subtoken_end),
                            'src_span': (src_subtoken_start, src_subtoken_end)
                        })

                        if ':inner:' in self._attention_regularization or ':all:' in self._attention_regularization:
                            # align_mat_view is wrapped by <s> and </s>
                            if 'complementary' in self._attention_regularization:
                                # alignment to outer tokens should be close to zero,
                                # while we do not pose any constraints to inner tokens
                                align_mat_view[
                                    tgt_subtoken_start: tgt_subtoken_end,
                                    src_subtoken_start: src_subtoken_end
                                ] = -1.0
                            else:
                                align_mat_view[
                                    tgt_subtoken_start: tgt_subtoken_end,
                                    src_subtoken_start: src_subtoken_end
                                ] = 1.0

                    if ':outer:' in self._attention_regularization or ':all:' in self._attention_regularization:
                        source_token_alignment_mask_for_sketch_token = np.ones(len(source_tokens))

                        if 'complementary' in self._attention_regularization:
                            source_token_alignment_mask_for_sketch_token.fill(-1)

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
                        source_span, (src_subtoken_start, src_subtoken_end) = self._get_subtokens_slice(
                            source_tokens,
                            (entry['source_token_idx'], entry['source_token_idx'] + 1), source_subtoken_offsets
                        )

                        target_span, (tgt_subtoken_start, tgt_subtoken_end) = self._get_subtokens_slice(
                            target_tokens[1:-1],  # account for <s> and </s>
                            (entry['target_token_idx'], entry['target_token_idx'] + 1), target_subtoken_offsets
                        )

                        align_mat_view[tgt_subtoken_start: tgt_subtoken_end, src_subtoken_start: src_subtoken_end] = 1.0

                for i in range(align_mat_view.shape[0]):
                    if sum(align_mat_view[i]) == 0:
                        align_mat_view[i] = alignment_padding_val

                align_matrix[0] = align_matrix[-1] = alignment_padding_val

                alignment_field = ArrayField(align_matrix, padding_value=alignment_padding_val)
                fields_dict['target_to_source_alignment'] = alignment_field
                meta_dict['alignment_info'] = alignment_info

        fields_dict['metadata'] = MetadataField(meta_dict)

        instance = Instance(fields_dict)

        return instance

    def _tokenize_source(
        self,
        source: Union[str, List[Token], List[str]],
    ):
        if isinstance(source, str):
            source_tokens_on_white_space = self._white_space_tokenizer.tokenize(source)
        elif isinstance(source, list):
            if isinstance(source[0], str):
                source_tokens_on_white_space = source
            else:
                assert isinstance(source[0], Token)
                source_tokens_on_white_space = [tok.text for tok in source]
        else:
            raise ValueError(source)

        source_tokens, source_subtoken_offsets = self._source_tokenizer.intra_word_tokenize(
            [tok.text for tok in source_tokens_on_white_space]
        )

        if self.dataset_name == 'cfq' and self._add_variable_names_to_vocab:
            # convert special token indices to [unusedxxx], therefore keeping the size of embeddings
            entity_pattern = re.compile(r'^m(\d+)$')
            token: Token
            for token in source_tokens:
                m = entity_pattern.match(token.text)
                if m:
                    token.text = token.text.upper()
                    token.text_id = self._source_tokenizer.tokenizer.convert_token_to_ids([f'[unused{m.group(1)}]'])[0]  # noqa
        elif self.dataset_name == 'text2sql' and self._add_variable_names_to_vocab:
            for token in source_tokens:
                if token.text in self._canonical_variable_index:
                    token_idx = self._canonical_variable_index[token.text]
                    token.text_id = self._source_tokenizer.tokenizer.convert_tokens_to_ids([f'[unused{token_idx}]'])[0]  # noqa

        return source_tokens, source_subtoken_offsets

    def _tokenize_target(
        self,
        target_string: str,
    ) -> Tuple[List[Token], Dict]:
        if self.dataset_name == 'calflow':
            return self._tokenize_sexp(target_string)
        elif self.dataset_name == 'cfq':
            return self._tokenize_sparql(target_string)
        elif self.dataset_name == 'text2sql':
            return self._tokenize_sql(target_string)
        else:
            raise RuntimeError(f'unknown dataset type {self.dataset_type}')

    def _tokenize_sparql(self, sparql_string):
        tokens = self._target_tokenizer.tokenize(sparql_string)
        source_span_mentions = []
        subtoken_offsets = []
        entity_pattern = re.compile(r'^M(\d+)$')
        for token_idx, token in enumerate(tokens):
            if self._cfq_add_variable_names_to_vocab and entity_pattern.match(token.text):
                source_span_mentions.append((token_idx, token_idx + 1))

            subtoken_offsets.append((token_idx, token_idx))

        metadata = {
            'source_span_mentions': source_span_mentions,
            'subtoken_offsets': subtoken_offsets
        }

        return tokens, metadata

    def _tokenize_sexp(
        self,
        target_sexp_string: str,
        source_tokens: Optional[List[Token]] = None
    ) -> Tuple[List[Token], Dict]:
        target_sexp = seq_to_sexp(target_sexp_string.split(' '))

        source_span_mentions = []
        subtoken_offsets: List[Tuple[int, int]] = []

        def sexp_to_tokenized_str(sexp: Sexp, start_idx=0) -> Tuple[List[Token], int]:
            if isinstance(sexp, list):
                if (
                    len(sexp) > 0 and
                    isinstance(sexp[0], str) and
                    sexp[0] in {'String', 'LocationKeyphrase', 'PersonName'}
                ):
                    is_empty_string = False
                    if len(sexp) == 2 and sexp[-1] == '""':
                        is_empty_string = True
                        string_subtokens = []
                        token_offsets = []
                        delimeter = '"'
                    else:
                        assert sexp[1] == sexp[-1] == '"', f'Error when parsing literals in {target_sexp_string}'
                        delimeter = sexp[1]

                        string_tokens = sexp[2:-1]
                        if self.use_pretrained_encoder:
                            string_subtokens, token_offsets = self._target_tokenizer._intra_word_tokenize(ensure_token_list(string_tokens))  # noqa
                            token_offsets = self._target_tokenizer._increment_offsets(token_offsets, start_idx + 3)  # noqa
                        else:
                            string_subtokens = self._target_tokenizer.tokenize(' '.join(string_tokens))
                            token_offsets = [(i + start_idx + 3, i + start_idx + 3) for i in range(len(string_tokens))]

                    sexp_tokens = [
                        Token('('),
                        Token(sexp[0]),
                        Token(delimeter)
                    ] + string_subtokens + [
                        Token(delimeter),
                        Token(')')
                    ]

                    subtoken_offsets.extend([
                        (start_idx, start_idx),
                        (start_idx + 1, start_idx + 1),
                        (start_idx + 2, start_idx + 2)
                    ] + token_offsets + [
                        (start_idx + 3 + len(string_subtokens), start_idx + 3 + len(string_subtokens)),
                        (start_idx + 3 + len(string_subtokens) + 1, start_idx + 3 + len(string_subtokens) + 1),
                    ])

                    source_span_mentions.append(
                        (start_idx + 3, start_idx + 3 + len(string_subtokens))
                    )

                    return sexp_tokens, start_idx + len(sexp_tokens)
                else:
                    sexp_tokens = [Token('(')]
                    subtoken_offsets.append((start_idx, start_idx))

                    child_end_idx = start_idx + 1
                    for child_node in sexp:
                        child_tokens, child_end_idx = sexp_to_tokenized_str(child_node, start_idx=child_end_idx)
                        sexp_tokens.extend(child_tokens)

                    sexp_tokens.append(Token(')'))
                    subtoken_offsets.append((start_idx + len(sexp_tokens) - 1, start_idx + len(sexp_tokens) - 1))

                    # for subtoken_offset, subtoken in enumerate(sexp_tokens):
                    #     subtoken_offsets.append((start_idx + subtoken_offset, start_idx + subtoken_offset))

                    return sexp_tokens, start_idx + len(sexp_tokens)
            else:
                subtoken_offsets.append((start_idx, start_idx))

                return [Token(sexp)], start_idx + 1

        tokenized_target_sexp, end_idx = sexp_to_tokenized_str(target_sexp)

        metadata = {
            'sexp': target_sexp,
            'source_span_mentions': source_span_mentions,
            'subtoken_offsets': subtoken_offsets
        }

        return tokenized_target_sexp, metadata

    def _tokenize_sql(self, sql_string):
        from utils.sql.sql_utils.text2sql_utils import CANONICAL_VARIABLES

        tokens = self._target_tokenizer.tokenize(sql_string)
        source_span_mentions = []
        subtoken_offsets = []
        for token_idx, token in enumerate(tokens):
            if token.text in CANONICAL_VARIABLES:
                source_span_mentions.append((token_idx, token_idx + 1))

            subtoken_offsets.append((token_idx, token_idx))

        metadata = {
            'source_span_mentions': source_span_mentions,
            'subtoken_offsets': subtoken_offsets
        }

        return tokens, metadata

    def _test_tokenized_target(
        self,
        target_sequence,
        target_subtokens,
        subtoken_offsets
    ):
        target_tokens = target_sequence.split(' ')
        for idx, token in enumerate(target_tokens):
            sub_tokens = target_subtokens[subtoken_offsets[idx][0]: subtoken_offsets[idx][1] + 1]
            sub_tokens = [tok.text for tok in sub_tokens]
            reconstructed_tgt_token = ' '.join(sub_tokens).replace(' ##', '')
            assert token.lower() == reconstructed_tgt_token.replace(' ', '').lower()

    @staticmethod
    def _get_subtokens_slice(
        tokens: List[Token],
        span: Tuple[int, int],
        tokens_offset: List[Optional[Tuple[int, int]]] = None
    ) -> Tuple[List[Token], Tuple[int, int]]:
        span_start, span_end = span
        if tokens_offset:
            subword_start = tokens_offset[span_start][0]
            subword_end = tokens_offset[span_end - 1][1] + 1

            tokens_slice = tokens[subword_start: subword_end]
            subword_span = (subword_start, subword_end)
        else:
            tokens_slice = tokens[span_start: span_end]
            subword_span = span

        return tokens_slice, subword_span

    @staticmethod
    def _tokens_to_first_appearing_indices(tokens: List[Union[Token, str]]) -> List[int]:
        """Convert tokens to first appearing indices in the sentence"""
        token_to_first_appearing_index_map: Dict[str, int] = {}
        out: List[int] = []

        for token in tokens:
            out.append(token_to_first_appearing_index_map.setdefault(
                token.text if isinstance(token, Token) else str(token), len(token_to_first_appearing_index_map))
            )

        return out


def main():
    reader = SequenceToSequenceModelWithCopyReader()
    dataset = reader.read('data/calflow.singleturn.top100.txt')
    vocab = Vocabulary.from_instances(dataset.instances)
    print(vocab)


if __name__ == '__main__':
    main()
