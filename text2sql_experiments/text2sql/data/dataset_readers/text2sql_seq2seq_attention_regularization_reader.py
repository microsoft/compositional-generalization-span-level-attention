from pathlib import Path
from typing import Dict, List
import logging
import json
import glob
import os
import sqlite3
import random
import numpy as np

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

import text2sql.data.dataset_readers.dataset_utils.text2sql_utils as text2sql_utils
from text2sql.data.preprocess.sql_templates import sql_schema_sanitize
from text2sql.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("text2sql_seq2seq_reader_att_reg")
class Seq2SeqDatasetReader(DatasetReader):
    def __init__(self,
                 schema_path: str,
                 database_path: str = None,
                 use_all_sql: bool = False,
                 use_all_queries: bool = True,
                 remove_unneeded_aliases: bool = False,
                 use_prelinked_entities: bool = True,
                 cross_validation_split_to_exclude: int = None,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False,
                 random_seed:int = 0,
                 schema_free_supervision=False,
                 attention_regularization: str = None
                 ) -> None:
        super().__init__(lazy)
        self._random_seed = random_seed
        self._source_tokenizer = source_tokenizer or WhitespaceTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token

        self._cross_validation_split_to_exclude = str(cross_validation_split_to_exclude)

        assert use_all_sql is False
        assert use_all_queries is True
        assert remove_unneeded_aliases is False
        assert schema_free_supervision is False
        assert use_prelinked_entities is True

        self._use_all_sql = use_all_sql
        self._use_all_queries = use_all_queries
        self._remove_unneeded_aliases = remove_unneeded_aliases
        self._use_prelinked_entities = use_prelinked_entities

        if database_path is not None:
            database_path = cached_path(database_path)
            connection = sqlite3.connect(database_path)
            self._cursor = connection.cursor()
        else:
            self._cursor = None

        self._schema_path = schema_path
        self._schema_free_supervision = schema_free_supervision
        self._attention_regularization = attention_regularization

    def _read(self, file_path: str):
        file_path = Path(file_path)

        with file_path.open('r') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")
                data = json.loads(line)

                source_sequence = data['source']
                target_sequence = data['target']
                tags = data['tags']
                alignment_info = data['alignments']

                token_level_alignment_info = data.get('token_level_alignments')

                yield self.text_to_instance(
                    source_sequence,
                    target_sequence,
                    alignment_info=alignment_info,
                    token_level_alignment_info=token_level_alignment_info,
                )

                # if line_num > 100:
                #     break

    @overrides
    def text_to_instance(
        self,
        source_string: str,
        target_string: str = None,
        alignment_info: List[Dict] = None,
        token_level_alignment_info: List[Dict] = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_source = self._source_tokenizer.tokenize(source_string)

        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))

        tokenized_source.append(Token(END_SYMBOL))

        source_field = TextField(tokenized_source, self._source_token_indexers)

        fields_dict = {
            'source_tokens': source_field
        }

        metadata = {
            'source_tokens': tokenized_source,
        }

        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            metadata['target_tokens'] = [tok.text for tok in tokenized_target]

            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))

            if self._attention_regularization:
                alignment_field = self.get_attention_regularization_fields(
                    tokenized_source, tokenized_target, alignment_info, token_level_alignment_info)
                fields_dict['target_to_source_alignment'] = alignment_field

            target_field = TextField(tokenized_target, self._target_token_indexers)
            fields_dict['target_tokens'] = target_field

        fields_dict['metadata'] = MetadataField(metadata)

        return Instance(fields_dict)

    def get_attention_regularization_fields(
        self,
        source_tokens: List[Token],
        target_tokens: List[Token],
        alignment_info: List[Dict],
        token_level_alignment_info: List[Dict]
    ) -> ArrayField:
        # `tokenized_target` is prepended/appended with <START> and <EOS> symbols

        use_token_level_regularization = ':token:' in self._attention_regularization
        use_segment_level_regularization = not use_token_level_regularization

        alignment_padding_val = -1
        align_matrix = np.zeros((len(target_tokens), len(source_tokens)))
        align_mat_view = align_matrix[1:-1]
        alignment_spans = []

        if use_segment_level_regularization and alignment_info:
            for entry in alignment_info:
                tgt_token_start, tgt_token_end = entry['target_tokens_idx']
                src_token_start, src_token_end = entry['source_tokens_idx']

                if self._source_add_start_token:
                    src_token_start, src_token_end = (src_token_start + 1, src_token_end + 1)

                alignment_spans.append({
                    'tgt_span': (tgt_token_start, tgt_token_end),
                    'src_span': (src_token_start, src_token_end)
                })

                if ':inner:' in self._attention_regularization or ':all:' in self._attention_regularization:
                    # wrapped by <s> and </s>
                    align_mat_view[
                        tgt_token_start: tgt_token_end,
                        src_token_start: src_token_end
                    ] = 1.0

            if ':outer:' in self._attention_regularization or ':all:' in self._attention_regularization:
                source_token_alignment_mask_for_sketch_token = np.ones(len(source_tokens))

                for alignment in alignment_spans:
                    source_token_alignment_mask_for_sketch_token[
                        alignment['src_span'][0]: alignment['src_span'][1]
                    ] = 0

                for tgt_token_idx in range(align_mat_view.shape[0]):
                    is_sketch_token = not (
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
                src_token_idx = entry['source_token_idx']
                tgt_token_idx = entry['target_token_idx']

                if self._source_add_start_token:
                    src_token_idx += 1

                align_mat_view[tgt_token_idx: tgt_token_idx + 1, src_token_idx: src_token_idx + 1] = 1.0

        for i in range(align_mat_view.shape[0]):
            if sum(align_mat_view[i]) == 0:
                align_mat_view[i] = alignment_padding_val

        align_matrix[0] = align_matrix[-1] = alignment_padding_val

        alignment_field = ArrayField(align_matrix, padding_value=alignment_padding_val)

        return alignment_field
