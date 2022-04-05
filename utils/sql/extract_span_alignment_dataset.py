import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Any, Tuple, Callable, Optional
import sqlparse
from allennlp.data import Token
from allennlp.data.tokenizers import SpacyTokenizer
from sqlparse import sql
from sqlparse.tokens import Punctuation, Keyword
from sqlparse.sql import Statement

from utils.sequence_utils import find_sequence
from utils.sql.sql_utils.encoder_input_canonicalizer import process_sentence
from utils.sql.sql_utils.text2sql_utils import SqlTokenizer, canonicalize_sql_for_alignment, process_sql_data_standard, \
    CANONICAL_VARIABLES

from utils.extract_giza_alignment import (
    get_giza_alignments,
    SpanAlignment,
    MergedSpanAlignment,
    construct_alignment
)
from utils.sql.sql_utils.text2sql_utils import SqlTokenizer

from utils.sql.string_util import STOP_WORDS


# Settings
CONSECUTIVE_UTTERANCE_SPAN = True


@dataclass(frozen=True)
class Source:
    tokens: List[str]


@dataclass(frozen=True)
class Target:
    value: str
    tokens: List[Token]
    to_original_offset_map: Dict


def is_stopword(word):
    return word.lower() in STOP_WORDS


def is_strictly_stopword(word):
    return word in {'the', 'thanks', 'please', "'re", 'would', 'is', 'are'}


def merge_source_span_alignments(alignments: List[SpanAlignment]) -> List[SpanAlignment]:
    alignments = list(alignments)
    while True:
        # sort by source index
        alignments = sorted(alignments, key=lambda a: a.source_span)
        can_merge = False

        for i in range(len(alignments) - 1):
            span = alignments[i]
            next_span = alignments[i + 1]

            dist = span.source_distance(next_span)
            if dist == 0:
                can_merge = True

                if can_merge:
                    merged_span = MergedSpanAlignment([span, next_span])

                    del alignments[i: i + 2]
                    alignments.append(merged_span)

                    break

        if not can_merge:
            break

    return alignments


def is_simple_sql_comparison_clause(clause: Union[sql.Token, sql.TokenList]):
    if isinstance(clause, sql.Comparison):
        if not isinstance(clause.right, sql.Parenthesis):
            return True

    return False


def post_processing(span_alignment_entries: List[Dict], target: Target) -> List[Dict]:
    city_name_and_alias_id_map = {}

    def _search_and_apply(_func: Callable):
        for _idx, _entry in enumerate(span_alignment_entries):
            _func(_entry, _idx)

    for entry in span_alignment_entries:
        target_start, target_span_end = entry['target_span_index']
        aligned_target_tokens = target.tokens[target_start: target_span_end]
        aligned_target = ' '.join([t.text for t in aligned_target_tokens])
        entry['target'] = aligned_target
        clause = entry['clause']

        # if simple comparision
        if is_simple_sql_comparison_clause(clause):
            left_clause = clause.left.value.strip()
            right_clause = clause.right.value.strip()

            if right_clause in {'city_name0', 'city_name1', 'city_name2', 'city_name3'}:
                entity_name = right_clause
                alias_id = int(re.search(r'CITYalias(\d)\.CITY_NAME', left_clause).group(1))
                city_name_and_alias_id_map[entity_name] = {
                    'alias_id': alias_id,
                    'flight_direction': None,
                    'alignment': entry
                }

    for city_name, entity_info in city_name_and_alias_id_map.items():
        from_to_airport_clause_alignment = []
        alias_id = entity_info['alias_id']

        def _func(_entry, _entry_idx):
            clause = _entry['clause']

            if is_simple_sql_comparison_clause(clause):
                clause_value = clause.value.strip()
                if (
                    f'FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE' in clause_value or
                    f'FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE' in clause_value
                ):
                    from_to_airport_clause_alignment.append(_entry)

        _search_and_apply(_func)

        if not from_to_airport_clause_alignment:
            continue

        from_to_airport_clause_alignment = from_to_airport_clause_alignment[-1]
        entity_alignment = entity_info['alignment']
        align1: SpanAlignment = from_to_airport_clause_alignment['alignment']
        align2: SpanAlignment = entity_alignment['alignment']

        segment_alignment = align2
        if align1.source_span != align2.source_span:
            merge = False
            if align1.source_overlap(align2):
                merge = True
            elif align1.source_span[1] == align2.source_span[0]:
                merge = True

            if merge:
                merged_alignment = MergedSpanAlignment([align1, align2])
                segment_alignment = merged_alignment

        entity_info['alignment'] = segment_alignment

        def _func(_entry, _entry_idx):
            clause = _entry['clause']

            if is_simple_sql_comparison_clause(clause):
                if clause.value.strip() in {
                    f'CITYalias{alias_id}.CITY_CODE = AIRPORT_SERVICEalias{alias_id}.CITY_CODE',
                    f'FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE',
                    f'FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE'
                }:
                    _entry['alignment'] = segment_alignment

        _search_and_apply(_func)

    return span_alignment_entries
    # for idx, entry in enumerate(span_alignment_entries):
    #     # Special case #1:
    #     # FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE should have the same alignment as
    #     # CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE
    #
    #     # m = re.search(r"FLIGHTalias(\d)\.FROM_AIRPORT = AIRPORT_SERVICEalias(\d)\.AIRPORT_CODE", entry['target'])
    #     # if m:
    #     #     alias_idx = int(m.group(1))
    #     #     tgt_clause = f"CITYalias{alias_idx}.CITY_CODE = AIRPORT_SERVICEalias{alias_idx}.CITY_CODE"
    #     #
    #     #     def _modifier(_other_entry: Dict, _other_entry_idx: int):
    #     #         if tgt_clause in _other_entry['target'] and idx != _other_entry_idx:
    #     #             entry['alignment'] = _other_entry['alignment']
    #     #
    #     #     _search_and_apply(_modifier)
    #     pass


def merge_adjacent_span_level_alignments(span_alignment_entries: List[Dict], target: Target) -> List[Dict]:
    continue_ = True
    merged_alignments = list(span_alignment_entries)
    while continue_:
        merged_alignments = sorted(merged_alignments, key=lambda a: a['target_span_index'])
        continue_ = False

        for idx in range(len(merged_alignments) - 1):
            entry = merged_alignments[idx]
            alignment: SpanAlignment = entry['alignment']
            target_start, target_span_end = entry['target_span_index']

            next_entry = merged_alignments[idx + 1]
            next_target_start, next_target_span_end = next_entry['target_span_index']
            next_alignment: SpanAlignment = next_entry['alignment']

            tokens_in_between = target.tokens[target_span_end: next_target_start]
            if (
                alignment.source_overlap(next_alignment) and
                all(token.text in {'AND', 'OR'} for token in tokens_in_between)
            ):
                merged_span_alignment = MergedSpanAlignment([alignment, next_alignment])
                del merged_alignments[idx: idx + 2]
                merged_alignments.append({
                    'alignment': merged_span_alignment,
                    'clause': [entry['clause'], next_entry['clause']],
                    'target_span_index': (target_start, next_target_span_end)
                })
                continue_ = True
                break
            elif (
                (target_start, target_span_end) == (next_target_start, next_target_span_end) and
                alignment.source_distance(next_alignment) == 0
            ):
                merged_span_alignment = MergedSpanAlignment([alignment, next_alignment])
                del merged_alignments[idx: idx + 2]
                merged_alignments.append({
                    'alignment': merged_span_alignment,
                    'clause': entry['clause'],
                    'target_span_index': (target_start, next_target_span_end)
                })
                continue_ = True
                break

    return merged_alignments


def get_span_index_from_char_offsets(
    target,
    tgt_span_char_offset: int,
    tgt_span_end_char_offset: int
) -> Tuple[int, int]:
    start_idx = [idx for idx, token in enumerate(target.tokens) if token.idx == tgt_span_char_offset][0]
    end_idx = [idx for idx, token in enumerate(target.tokens) if token.idx_end == tgt_span_end_char_offset][0] + 1

    return start_idx, end_idx


def find_aligned_source_spans(
    target_span_start_idx: int,
    target_span_end_idx: int,
    span_alignments: List,
    existing_alignment_results: List,
    source, target,
    allow_source_overlap: bool = True,
    allowed_source_tokens: Optional[List[str]] = None,
    consecutive_utterance_span: Optional[bool] = None
):
    if consecutive_utterance_span is None:
        global CONSECUTIVE_UTTERANCE_SPAN
        consecutive_utterance_span = CONSECUTIVE_UTTERANCE_SPAN

    # find overlapping SpanAlignments
    overlapped_alignments = [
        alignment
        for alignment in span_alignments
        if (
            (
                target_span_start_idx <= alignment.target_span[0] <= target_span_end_idx - 1 or
                target_span_start_idx <= alignment.target_span[1] - 1 <= target_span_end_idx - 1 or
                alignment.target_span[0] <= target_span_start_idx <= target_span_end_idx - 1 <= alignment.target_span[1] - 1
            ) and (
                allow_source_overlap or not any(
                    alignment.source_overlap(m['alignment'])
                    for m in existing_alignment_results
                    if m['alignment'] is not None
                )
            )
        )
    ]

    # remove alignments of punctuations
    filtered_alignments = []
    for alignment in overlapped_alignments:
        aligned_source = ' '.join(source.tokens[alignment.source_span[0]: alignment.source_span[1]])
        aligned_target = ' '.join([t.text for t in target.tokens[alignment.target_span[0]: alignment.target_span[1]]])

        is_valid = True
        if aligned_target in {'(', ')', '='}:
            is_valid = False

        if allowed_source_tokens and aligned_source not in allowed_source_tokens:
            is_valid = False
        # if ('DAYSalias' in aligned_target or 'DATE_DAYalias' in aligned_target) and 'city_name' in aligned_source:
        #     is_valid = False

        if is_valid:
            filtered_alignments.append(alignment)

    overlapped_alignments = filtered_alignments

    if not overlapped_alignments:
        return []

    overlapped_alignments.sort(key=lambda span: span.source_span)

    while True:
        # sort by source index
        overlapped_alignments = sorted(overlapped_alignments, key=lambda a: (a.source_span[0], a.target_span[0]))
        can_merge = False

        for i in range(len(overlapped_alignments) - 1):
            span = overlapped_alignments[i]
            next_span = overlapped_alignments[i + 1]

            dist = span.distance(next_span)
            if dist == 0:
                can_merge = True

                if can_merge:
                    merged_span = MergedSpanAlignment([span, next_span])

                    del overlapped_alignments[i: i + 2]
                    overlapped_alignments.append(merged_span)

                    break

        if not can_merge:
            break

    alignments = overlapped_alignments

    # fill the gaps if the gap is composed of stop words
    alignments.sort(key=lambda a: a.source_span)
    for idx in range(len(overlapped_alignments) - 1):
        alignment = overlapped_alignments[idx]
        next_alignment = overlapped_alignments[idx + 1]

        all_stopwords = all(
            is_stopword(source.tokens[src_idx])
            for src_idx
            in range(alignment.source_span[1], next_alignment.source_span[0])
        )

        if all_stopwords:
            overlapped_alignments.append(
                SpanAlignment(
                    (alignment.source_span[1], next_alignment.source_span[0]),
                    (target_span_start_idx, target_span_end_idx)
                )
            )

    filtered_alignments = []
    for alignment in alignments:
        aligned_source = ' '.join(source.tokens[alignment.source_span[0]: alignment.source_span[1]])
        aligned_target = ' '.join([t.text for t in target.tokens[alignment.target_span[0]: alignment.target_span[1]]])

        is_valid = True
        # if ('DAYSalias' in aligned_target or 'DATE_DAYalias' in aligned_target) and 'city_name' in aligned_source:
        #     is_valid = False

        if is_valid:
            filtered_alignments.append(alignment)

    alignments = merge_source_span_alignments(filtered_alignments)
    alignments = [
        alignment
        for alignment
        in alignments
        if not is_strictly_stopword(
            ' '.join(
                source.tokens[
                    alignment.source_span[0]:
                    alignment.source_span[1]
                ]
            )
        )
    ]

    if consecutive_utterance_span and alignments:
        alignments = [MergedSpanAlignment(alignments)]

    if not allow_source_overlap:
        raise NotImplementedError

    return alignments


def find_and_add_aligned_source_spans(
    clause: Union[sql.TokenList, List[sql.Token]],
    target_span_start_idx: int,
    target_span_end_idx: int,
    span_alignments: List,
    alignment_results: List,
    source: Source, target: Target,
    aligned_source_tokens: List[str] = None,
    allowed_source_tokens: List[str] = None
):
    if aligned_source_tokens:
        alignments = []
        aligned_source_tokens = sorted(aligned_source_tokens, key=lambda t: len(t.split(' ')), reverse=True)
        for token_string in aligned_source_tokens:
            tokens = token_string.split(' ')
            try:
                token_idx = find_sequence(source.tokens, tokens)
                alignment = SpanAlignment(
                    token_idx,
                    (target_span_start_idx, target_span_end_idx)
                )
                if not any(alignment.source_overlap(a) for a in alignments):
                    alignments.append(alignment)
            except IndexError:
                continue

        clause_alignments = merge_source_span_alignments(alignments)
    else:
        clause_alignments = find_aligned_source_spans(
            target_span_start_idx, target_span_end_idx,
            span_alignments,
            alignment_results,
            source, target,
            allowed_source_tokens=allowed_source_tokens
        )

    for alignment in clause_alignments:
        alignment_results.append({
            'clause': clause,
            'target_span_index': (target_span_start_idx, target_span_end_idx),
            'alignment': alignment
        })


def parse_where_clause(
    clause: Union[sql.Where, sql.Parenthesis],
    char_offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    ptr = 0
    tokens: List[sql.Token] = clause.tokens
    while ptr < len(tokens):
        token = tokens[ptr]
        update_offset = True

        if token.is_whitespace or token.normalized in ('WHERE', 'OR', 'AND', '(', ')'):
            pass
        elif isinstance(token, sql.Comparison):
            parse_comparison(
                token,
                char_offset,
                source,
                target,
                span_alignments,
                alignment_results
            )
        elif isinstance(token, sql.Parenthesis):
            parse_where_clause(
                token, char_offset,
                source, target,
                span_alignments,
                alignment_results
            )
        elif isinstance(token, sql.Identifier):
            # deal with special case of <id> BETWEEN <id|num> AND <id|num>
            if first_non_stopword(tokens[ptr + 1:]).normalized == 'BETWEEN':
                between_end_idx = first_non_stopword(tokens[ptr:], nth=5, index_only=True)
                char_offset = parse_between_and(
                    tokens[ptr:ptr + between_end_idx + 1],
                    char_offset,
                    source, target,
                    span_alignments, alignment_results
                )
                ptr += between_end_idx + 1
                update_offset = False
            elif (
                first_non_stopword(tokens[ptr + 1:]).normalized == 'NOT' and
                first_non_stopword(tokens[ptr + 1:], nth=2).normalized == 'BETWEEN'
            ):
                between_end_idx = first_non_stopword(tokens[ptr:], nth=6, index_only=True)
                char_offset = parse_between_and(
                    tokens[ptr:ptr + between_end_idx + 1],
                    char_offset,
                    source, target,
                    span_alignments, alignment_results
                )
                ptr += between_end_idx + 1
                update_offset = False
            elif first_non_stopword(tokens[ptr + 1:]).normalized == 'IS':
                # <id> IS NULL | <is> IS NOT NULL
                allowing_delimeters = ['AND', 'OR', 'GROUP', 'ORDER', 'LIMIT', ')']
                try:
                    clause_end_index = first_non_stopword(
                        tokens[ptr:],
                        matcher=lambda tok: not tok.is_whitespace and tok.value in allowing_delimeters,
                        index_only=True
                    )
                except IndexError:
                    clause_end_index = len(tokens[ptr:])

                while tokens[ptr + clause_end_index - 1].is_whitespace:
                    clause_end_index -= 1

                char_offset = parse_sql_tokens_span(
                    tokens[ptr:ptr + clause_end_index],
                    char_offset,
                    source, target,
                    span_alignments, alignment_results
                )
                ptr += clause_end_index
                update_offset = False
            else:
                raise ValueError(token)

        if update_offset:
            char_offset += len(token.value)
            ptr += 1

    return char_offset


def first_non_stopword(
    tokens: List[sql.Token], *,
    matcher: Callable = None, nth: int = 1, index_only: bool = False
) -> Union[sql.Token, int]:
    if not matcher:
        matcher = lambda _token: not (_token.is_whitespace or _token.ttype == Punctuation)

    ptr = 0
    for idx, token in enumerate(tokens):
        if matcher(token):
            ptr += 1
            if ptr == nth:
                return (idx if index_only else token)

    raise IndexError


def parse_comparison(
    clause: sql.Comparison,
    char_offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict],
    use_heuristics: bool = False,
    analyze_nested_query: bool = False
):
    if isinstance(clause.right, sql.Parenthesis):
        # nested SQL query
        if first_non_stopword(clause.right).normalized == 'SELECT':
            if analyze_nested_query:
                # parse the left operand
                parse_sql_tokens_span(
                    clause.left,
                    char_offset,
                    source, target,
                    span_alignments, alignment_results
                )

                right_char_offset = char_offset
                for token in clause.tokens[:-1]:
                    right_char_offset += len(token.value)
                right_char_offset += len(clause.right.tokens[0].value)
                current_alignment_results_num = len(alignment_results)
                parse_stmt(
                    clause.right.tokens[1:-1],
                    right_char_offset,
                    source, target,
                    span_alignments, alignment_results
                )

                # new_alignment_results_num = len(alignment_results)

                # align the left operand to the first alignment result of the right operand
                # if new_alignment_results_num > current_alignment_results_num:
                #     right_first_alignment = alignment_results[current_alignment_results_num]
                #     left_op_target_span_start_idx, left_op_target_span_end_idx = get_span_index_from_char_offsets(
                #         target, char_offset, char_offset + len(clause.left.value))
                #     alignment_results.append({
                #         'clause': clause.left,
                #         'target_span_index': (left_op_target_span_start_idx, left_op_target_span_end_idx),
                #         'alignment': SpanAlignment(
                #             (right_first_alignment['alignment'].source_span),
                #             (left_op_target_span_start_idx, left_op_target_span_end_idx)
                #         )
                #     })
            else:
                target_span_start_idx, target_span_end_idx = get_span_index_from_char_offsets(
                    target, char_offset, char_offset + len(clause.value))

                find_and_add_aligned_source_spans(
                    clause,
                    target_span_start_idx, target_span_end_idx,
                    span_alignments,
                    alignment_results,
                    source, target
                )
        else:
            raise ValueError(clause.right)
    else:
        target_span_start_idx, target_span_end_idx = get_span_index_from_char_offsets(
            target, char_offset, char_offset + len(clause.value))

        if use_heuristics:
            if is_simple_sql_comparison_clause(clause):
                clause_value = clause.value.strip()
                aligned_variables = [
                    var
                    for var
                    in {
                        'city_name0',
                        'city_name1',
                        'city_name2',
                        'state_name0',
                        'state_name1',
                        'state_name2',
                    }
                    if var in clause_value
                ]
                if aligned_variables:
                    find_and_add_aligned_source_spans(
                        clause,
                        target_span_start_idx, target_span_end_idx,
                        span_alignments,
                        alignment_results,
                        source, target,
                        aligned_source_tokens=aligned_variables
                    )
                elif (
                    'FROM_AIRPORT' in clause_value
                ):
                    find_and_add_aligned_source_spans(
                        clause,
                        target_span_start_idx, target_span_end_idx,
                        span_alignments,
                        alignment_results,
                        source, target,
                        allowed_source_tokens=[
                            'from', 'leaving from', 'leaving', 'leave', 'departing from', 'departing',
                        ]
                    )
                elif (
                    'TO_AIRPORT' in clause_value
                ):
                    find_and_add_aligned_source_spans(
                        clause,
                        target_span_start_idx, target_span_end_idx,
                        span_alignments,
                        alignment_results,
                        source, target,
                        allowed_source_tokens=[
                            'to', 'arriving in', 'arrive', 'returning to'
                        ]
                    )
                elif not any(
                    clause_value in
                    {
                        f'CITYalias{alias_id}.CITY_CODE = AIRPORT_SERVICEalias{alias_id}.CITY_CODE',
                        f'FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE',
                        f'FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE',
                        f'FLIGHTalias1.FROM_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE',
                        f'FLIGHTalias1.TO_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE',
                        f'FLIGHTalias2.FROM_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE',
                        f'FLIGHTalias2.TO_AIRPORT = AIRPORT_SERVICEalias{alias_id}.AIRPORT_CODE',
                        f'STATEalias{alias_id}.STATE_CODE = CITYalias{alias_id}.STATE_CODE',
                    }
                    for alias_id in range(0, 8)
                ):
                    find_and_add_aligned_source_spans(
                        clause,
                        target_span_start_idx, target_span_end_idx,
                        span_alignments,
                        alignment_results,
                        source, target
                    )
        else:
            find_and_add_aligned_source_spans(
                clause,
                target_span_start_idx, target_span_end_idx,
                span_alignments,
                alignment_results,
                source, target
            )

    char_offset += len(clause.value)

    return char_offset


def parse_between_and(
    tokens: List[sql.Token],
    char_offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    clause_end_char_offset = char_offset + sum(len(token.value) for token in tokens)

    target_span_start_idx, target_span_end_idx = get_span_index_from_char_offsets(
        target, char_offset, clause_end_char_offset)

    find_and_add_aligned_source_spans(
        tokens,
        target_span_start_idx, target_span_end_idx,
        span_alignments, alignment_results,
        source, target
    )

    return clause_end_char_offset


def parse_sql_tokens_span(
    tokens: List[sql.Token],
    char_offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    clause_end_char_offset = char_offset + sum(len(token.value) for token in tokens)

    target_span_start_idx, target_span_end_idx = get_span_index_from_char_offsets(
        target, char_offset, clause_end_char_offset)

    find_and_add_aligned_source_spans(
        tokens,
        target_span_start_idx, target_span_end_idx,
        span_alignments, alignment_results,
        source, target
    )

    return clause_end_char_offset


def parse_group_by_clause(
    tokens: List[sql.Token],
    offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    ptr = 0
    while ptr < len(tokens) and (
        tokens[ptr].is_whitespace or
        isinstance(tokens[ptr], (sql.Identifier, sql.IdentifierList)) or
        tokens[ptr].normalized in {'GROUP BY', 'HAVING'}
    ):
        offset += len(tokens[ptr].value)
        ptr += 1

    return offset


def parse_order_by_clause(
    tokens: List[sql.Token],
    offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    start_offset = offset
    ptr = 0
    while ptr < len(tokens) and (
        tokens[ptr].is_whitespace or
        isinstance(tokens[ptr], (sql.Identifier, sql.IdentifierList)) or
        tokens[ptr].normalized in {'ORDER BY', 'ASC', 'DESC', 'LIMIT'}
    ):
        offset += len(tokens[ptr].value)
        ptr += 1

    char_offset = parse_sql_tokens_span(
        tokens[0:ptr],
        start_offset,
        source, target,
        span_alignments, alignment_results
    )

    return char_offset


def parse_identifier(
    clause: sql.Identifier,
    char_offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    return parse_identifier_list([clause], char_offset, source, target, span_alignments, alignment_results)


def parse_identifier_list(
    clause: Union[sql.IdentifierList, List[sql.Token]],
    char_offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    tokens = clause if isinstance(clause, list) else clause.tokens
    for token in tokens:
        if isinstance(token, (sql.Function, sql.Identifier)):
            target_span_start_idx, target_span_end_idx = get_span_index_from_char_offsets(
                target, char_offset, char_offset + len(token.value))

            find_and_add_aligned_source_spans(
                token,
                target_span_start_idx, target_span_end_idx,
                span_alignments,
                alignment_results,
                source, target
            )

        char_offset += len(token.value)

    return char_offset


def get_token_idx_at_offset(tokens: List[sql.Token], start_offset, offset: int) -> int:
    ptr = start_offset
    for idx, token in enumerate(tokens):
        if offset == ptr:
            return idx

        ptr += len(token.value)

    # end of clause
    if offset == ptr:
        return len(tokens)
    else:
        raise IndexError


def parse_stmt(
    clause: [sql.Statement, List[sql.Token]],
    char_offset: int,
    source: Source,
    target: Target,
    span_alignments: List[Dict],
    alignment_results: List[Dict]
):
    token: sql.Token
    ptr = 0
    start_offset = char_offset
    tokens = clause if isinstance(clause, list) else clause.tokens
    while ptr < len(tokens):
        token = tokens[ptr]
        update_offset = True

        if token.is_whitespace or token.normalized in {'SELECT', 'DISTINCT'}:
            pass
        elif token.normalized in {'MIN', 'MAX', 'AVG', 'COUNT'}:
            column_end_idx = first_non_stopword(tokens[ptr:], index_only=True, nth=2)
            # MAX ( Identifier ) | COUNT DISTINCT xxx
            char_offset = parse_sql_tokens_span(
                tokens[ptr:ptr + column_end_idx + 1], char_offset,
                source, target,
                span_alignments, alignment_results
            )
            ptr += column_end_idx + 1
            update_offset = False
        elif isinstance(token, sql.Identifier):
            parse_identifier(token, char_offset, source, target, span_alignments, alignment_results)
        elif isinstance(token, sql.IdentifierList):
            parse_identifier_list(token.tokens, char_offset, source, target, span_alignments, alignment_results)
        elif isinstance(token, sql.Comparison):
            # SELECT { COUNT (*) > 1 }
            parse_comparison(token, char_offset, source, target, span_alignments, alignment_results)
        elif token.normalized == 'FROM':
            # skip identifier list in FROM clause
            char_offset += len(token.value)
            ptr += 1

            while ptr < len(tokens) and (
                tokens[ptr].is_whitespace or
                isinstance(tokens[ptr], (sql.IdentifierList, sql.Identifier))
            ):
                char_offset += len(tokens[ptr].value)
                ptr += 1

            update_offset = False
        elif isinstance(token, sql.Where):
            parse_where_clause(token, char_offset, source, target, span_alignments, alignment_results)
        elif token.normalized == 'GROUP BY':
            char_offset = parse_group_by_clause(tokens[ptr:], char_offset, source, target, span_alignments, alignment_results)
            update_offset = False
        elif token.normalized == 'ORDER BY':
            char_offset = parse_order_by_clause(tokens[ptr:], char_offset, source, target, span_alignments, alignment_results)
            update_offset = False
        elif token.normalized in 'ASC':
            raise ValueError(token)

        if update_offset:
            char_offset += len(token.value)
            ptr += 1
        else:
            ptr = get_token_idx_at_offset(tokens, start_offset, char_offset)

    return char_offset


whitespace_tokenizer = SpacyTokenizer(split_on_spaces=True)


def compute_span_level_alignments(
    source: str,
    target: str,
    simplified_sql_for_alignment: str,
    simplified_sql_to_target_sql_token_offset: List[Tuple[int, int]],
    aligner_outputs: List[List],
    debug: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    token_alignments = []
    for src_idx, src_tok_align in enumerate(aligner_outputs):
        if src_tok_align:
            if not isinstance(src_tok_align, list):
                src_tok_align = [src_tok_align]

            for tgt_idx in src_tok_align:
                token_alignments.append((src_idx, tgt_idx))

    candidate_span_alignments = construct_alignment(None, None, token_alignments, merge=False)

    source_tokens = source.split(' ')
    source = Source(source_tokens)

    # tokenize SQL
    target_tokens = target.split(' ')

    aligned_sql_tokens = whitespace_tokenizer.tokenize(simplified_sql_for_alignment)
    aligned_target = Target(simplified_sql_for_alignment, aligned_sql_tokens, to_original_offset_map=simplified_sql_to_target_sql_token_offset)
    parse_result: Statement = sqlparse.parse(simplified_sql_for_alignment)[0]  # only one statement

    alignment_results = []
    parse_stmt(parse_result, 0, source, aligned_target, candidate_span_alignments, alignment_results)

    # merge adjacent alignments?
    # alignment_results = post_processing(alignment_results, aligned_target)
    # alignment_results = merge_adjacent_span_level_alignments(alignment_results, aligned_target)

    alignment_entries = []
    for entry in alignment_results:
        alignment = entry['alignment']
        source_span = alignment.source_span
        clause = entry['clause']

        target_start, target_span_end = entry['target_span_index']
        target_sql_span_start_idx = simplified_sql_to_target_sql_token_offset[target_start][0]
        target_sql_span_end_idx = simplified_sql_to_target_sql_token_offset[target_span_end - 1][1] + 1

        alignment_entry = {
            'target_tokens_idx': (target_sql_span_start_idx, target_sql_span_end_idx),
            'simplified_target_tokens_idx': (target_start, target_span_end),
            'target_tokens': [t for t in target_tokens[target_sql_span_start_idx: target_sql_span_end_idx]],
            'simplified_target_tokens': [t.text for t in aligned_sql_tokens[target_start: target_span_end]],
            'source_tokens': source_tokens[alignment.source_span[0]: alignment.source_span[1]],
            'source_tokens_idx': tuple(alignment.source_span)
        }

        if debug:
            print(f"{' '.join(alignment_entry['simplified_target_tokens'])} ---> {source.tokens[source_span[0]: source_span[1]]}")
            print('***')

        alignment_entries.append(alignment_entry)

    # extract token level alignments
    token_level_alignments = []
    for src_token_idx, tgt_for_align_token_idx in token_alignments:
        target_token_idx = simplified_sql_to_target_sql_token_offset[tgt_for_align_token_idx][0]
        alignment_entry = {
            'source_token_idx': src_token_idx,
            'source_token': source_tokens[src_token_idx],
            'target_token_idx': target_token_idx,
            'target_token': target_tokens[target_token_idx]
        }
        token_level_alignments.append(alignment_entry)

    return alignment_entries, token_level_alignments
    # parsed_token_ptr = 0
    # canonical_token_ptr = 0
    # # result columns clause: SELECT OP(select_tables) FROM
    # while True:
    #     assert parse_result.tokens[parsed_token_ptr].normalized == 'SELECT'
    #
    #     result_column_clause_start_idx = canonical_token_ptr
    #     parsed_token_ptr, canonical_token_ptr = inc_until
    #     while parse_result.tokens[parsed_token_ptr]:
    #         parsed_token_ptr += 1
    #         if not parse_result.tokens[parsed_token_ptr].is_whitespace:
    #             canonical_token_ptr += 1
    #
    #     result_column_clause_end_idx = canonical_token_ptr - 1
    #
    #
    #     # inc to `WHERE`
    #     while not isinstance(parse_result.tokens[parsed_token_ptr], sql.Where):
    #         parsed_token_ptr += 1
    #         if not parse_result.tokens[parsed_token_ptr].is_whitespace:
    #             canonical_token_ptr += 1
    #
    #     where_clause: sql.Where = parse_result.tokens[parsed_token_ptr]
    #     parsed_token_ptr = 0
    #     while True:
    #         parsed_token_ptr += 1
    #         token: sql.Token = where_clause.tokens[parsed_token_ptr]
    #
    #         if not token.is_whitespace:
    #             canonical_token_ptr += 1
    #
    #         if isinstance(token, sql.Comparison):
    #             clause_start_idx = canonical_token_ptr
    #             clause_end_idx = _inc_token_to_caluse_end(canonical_token_ptr, token)
    #             get_alignment_for_sql_clause(
    #                 (clause_start_idx, clause_end_idx),
    #                 alignments, source_tokens, sql_tokens
    #             )
    #         elif isinstance(token, sql.Parenthesis):
    #             clause_start_idx = canonical_token_ptr
    #             clause_end_idx = _inc_token_to_caluse_end(canonical_token_ptr, token)
    #             parse_where_clause(token)
    #         elif token.normalized == 'GROUP BY':
    #             offset = parse_group_by_clause()


def generate_dataset(
    dataset_file: Path,
    output_file: Path,
    alignment_file_prefix: str = None,
    compute_alignment: bool = False,
):
    sql_tokenizer = SqlTokenizer()

    if compute_alignment:
        aligner_results = get_giza_alignments(
            dataset_file.with_name(f'{alignment_file_prefix}_s2t.VA3.final'),
            dataset_file.with_name(f'{alignment_file_prefix}_t2s.VA3.final'),
            output=dataset_file.with_suffix('.alignment.json')
        )

        alignment_meta_data = [
            json.loads(line) for line in
            dataset_file.with_suffix('.alignment_meta.jsonl').open()
        ]

    dataset = json.load(dataset_file.open())

    raw_examples: List[Tuple[str, str]] = list(process_sql_data_standard(
        dataset,
        use_linked=True,
        use_all_queries=True,
        use_all_sql=False
    ))

    examples = []

    for idx, (source, sql_query) in enumerate(raw_examples):
        # if idx > 500:
        #     continue

        sql_tokens = sql_tokenizer.tokenize(sql_query)
        target_sql = ' '.join(sql_tokens)

        canonical_sql_tokens, sql_token_offsets = canonicalize_sql_for_alignment(sql_tokens)
        canonical_sql_query = ' '.join(canonical_sql_tokens)

        canonical_source = process_sentence(source)
        source_tokens = source.split(' ')

        span_alignments = []
        token_alignments = []
        if compute_alignment:
            example_aligner_result = aligner_results[idx]
            example_alignment_metadata = alignment_meta_data[idx]

            if (
                example_aligner_result['is_valid_alignment'] and
                # example_aligner_result['example_idx'] not in {140, 1165, 1565, 4229, 4325} and
                'ALL' not in example_aligner_result['target_tokens']
            ):
                source_for_alignment = ' '.join(example_aligner_result['source_tokens'])
                simplified_target_for_alignment = ' '.join(example_aligner_result['target_tokens'])

                assert canonical_source == source_for_alignment
                is_valid_sql_for_alignment = True
                if canonical_sql_query != simplified_target_for_alignment:
                    print('SQL for alignment is invalid!')
                    is_valid_sql_for_alignment = False

                if is_valid_sql_for_alignment:
                    print(f'Example Idx: {idx}')
                    print(f'Source: {source}')
                    print(f'Target: {target_sql}')
                    print(f'Alignments:')
                    span_alignments, token_alignments = compute_span_level_alignments(
                        source, target_sql,
                        simplified_target_for_alignment,
                        example_alignment_metadata['simplified_sql_for_alignment_to_original_token_offset'],
                        example_aligner_result['s2t_alignment'],
                        debug=True
                    )

        example = {
            'example_idx': idx,
            'source': canonical_source,
            'target': target_sql,
            'tags': [],
            'alignments': span_alignments,
            'token_level_alignments': token_alignments
        }

        examples.append(example)

    with output_file.open('w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def test_alignment(

):
    dataset_root = Path('/Users/pengcheng/Research/improving-compgen-in-semparse/data/sql_data/')
    dataset_path = dataset_root / 'atis' / 'schema_full_split'
    aligner_results = get_giza_alignments(
        dataset_path / 's2t.VA3.final',
        dataset_path / 't2s.VA3.final',
    )

    # test_idx = 2

    # example = aligner_results[test_idx]
    # source = ' '.join(example['source_tokens'])
    # simplified_target = ' '.join(example['target_tokens'])
    #
    # compute_span_level_alignments(
    #     source, simplified_target,
    #     example['s2t_alignment']
    # )

    for example in aligner_results:
        if example['is_valid_alignment'] and example['example_idx'] not in {140, 1165, 1565, 4229, 4325} and 'ALL' not in example['target_tokens']:
            source = ' '.join(example['source_tokens'])
            simplified_target = ' '.join(example['target_tokens'])

            compute_span_level_alignments(
                source, simplified_target,
                example['s2t_alignment']
            )


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
def debug():
    # noqa
    sql = """
SELECT DISTINCT FLIGHTalias0.FLIGHT_ID FROM Student AS T1, Family as T2 WHERE ( 
        ( 
            CITYalias1.CITY_CODE = AIRPORT_SERVICEalias1.CITY_CODE AND 
            CITYalias1.CITY_NAME = city_name0 AND 
            FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE AND 
            FOOD_SERVICEalias0.MEAL_CODE = FLIGHTalias0.MEAL_CODE AND 
            FOOD_SERVICEalias0.MEAL_DESCRIPTION = meal_description0 
        ) 
        AND CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = city_name1 AND FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE 
    ) AND FLIGHTalias0.DEPARTURE_TIME = (
        SELECT MIN ( FLIGHTalias1.DEPARTURE_TIME )
        FROM WHERE ( 
            CITYalias3.CITY_CODE = AIRPORT_SERVICEalias3.CITY_CODE AND 
            CITYalias3.CITY_NAME = city_name0 AND 
            FLIGHTalias1.TO_AIRPORT = AIRPORT_SERVICEalias3.AIRPORT_CODE AND 
            FOOD_SERVICEalias1.MEAL_CODE = FLIGHTalias1.MEAL_CODE AND 
            FOOD_SERVICEalias1.MEAL_DESCRIPTION = meal_description0 
        ) AND CITYalias2.CITY_CODE = AIRPORT_SERVICEalias2.CITY_CODE AND CITYalias2.CITY_NAME = city_name1 AND FLIGHTalias1.FROM_AIRPORT = AIRPORT_SERVICEalias2.AIRPORT_CODE 
    )
"""

    sql = """
SELECT school, teach 
FROM SCHOOL_TABLE AS T1, TEACHER_TABLE AS T2 
WHERE 
school < 123 + 124 AND 
teach > ( SELECT MIN(teach) FROM TEACHER_TABLE  ) AND
teach IS NOT NULL
"""

# i would like an afternoon flight from city_name0 state_name0 to city_name1 state_name1

    sql = """
SELECT 
 DISTINCT FLIGHTalias0 . FLIGHT_ID 
FROM 
  AIRPORT_SERVICE AS AIRPORT_SERVICEalias0 , 
  AIRPORT_SERVICE AS AIRPORT_SERVICEalias1 , 
  CITY AS CITYalias0 , 
  CITY AS CITYalias1 , 
  FLIGHT AS FLIGHTalias0 , 
  STATE AS STATEalias0 , 
  STATE AS STATEalias1 
WHERE ( 
  CITYalias0 . CITY_CODE = AIRPORT_SERVICEalias0 . CITY_CODE AND 
  CITYalias0 . CITY_NAME = "city_name0" AND 
  CITYalias1 . CITY_CODE = AIRPORT_SERVICEalias1 . CITY_CODE AND 
  CITYalias1 . CITY_NAME = "city_name1" AND 
  FLIGHTalias0 . FROM_AIRPORT = AIRPORT_SERVICEalias0 . AIRPORT_CODE AND 
  FLIGHTalias0 . TO_AIRPORT = AIRPORT_SERVICEalias1 . AIRPORT_CODE AND 
  STATEalias0 . STATE_CODE = CITYalias0 . STATE_CODE AND 
  STATEalias0 . STATE_NAME = "state_name0" AND 
  STATEalias1 . STATE_CODE = CITYalias1 . STATE_CODE AND 
  STATEalias1 . STATE_NAME = "state_name1" 
  ) AND 
  FLIGHTalias0 . DEPARTURE_TIME BETWEEN departure_time0 AND departure_time1
"""

    # what 're the cheapest nonstop flights from city_name0 to city_name1 1 way
    sql = """
    SELECT 
        -- flights
        DISTINCT FLIGHTalias0 . FLIGHT_ID 
    FROM 
        AIRPORT_SERVICE AS AIRPORT_SERVICEalias0 , 
        AIRPORT_SERVICE AS AIRPORT_SERVICEalias1 , 
        CITY AS CITYalias0 , 
        CITY AS CITYalias1 , 
        FARE AS FAREalias0 , 
        FLIGHT AS FLIGHTalias0 , 
        FLIGHT_FARE AS FLIGHT_FAREalias0 
    WHERE 
        ( 
            ( 
                -- from cityname0 (actual: flights from cityname0)
                CITYalias0 . CITY_CODE = AIRPORT_SERVICEalias0 . CITY_CODE AND CITYalias0 . CITY_NAME = \" city_name0 \" AND
                -- to cityname1 (actual: cityname1)
                CITYalias1 . CITY_CODE = AIRPORT_SERVICEalias1 . CITY_CODE AND CITYalias1 . CITY_NAME = \" city_name1 \" AND 
                -- from cityname0 (actual: cityname1)
                FLIGHTalias0 . FROM_AIRPORT = AIRPORT_SERVICEalias0 . AIRPORT_CODE AND
                -- to city_name1 (actual: cityname1)
                FLIGHTalias0 . TO_AIRPORT = AIRPORT_SERVICEalias1 . AIRPORT_CODE 
            ) AND 
            -- nonstop
            FLIGHTalias0 . STOPS = stops0 
        ) AND 
        FAREalias0 . ONE_DIRECTION_COST = ( 
            SELECT 
                -- cheapest (actual: the cheapest nonstop)
                MIN ( FAREalias1 . ONE_DIRECTION_COST ) 
            FROM 
                AIRPORT_SERVICE AS AIRPORT_SERVICEalias2 , 
                AIRPORT_SERVICE AS AIRPORT_SERVICEalias3 , 
                CITY AS CITYalias2 , CITY AS CITYalias3 , 
                FARE AS FAREalias1 , 
                FLIGHT AS FLIGHTalias1 , 
                FLIGHT_FARE AS FLIGHT_FAREalias1 
            WHERE ( 
                    -- from cityname0 (actual: cheapest)
                    CITYalias2 . CITY_CODE = AIRPORT_SERVICEalias2 . CITY_CODE AND
                     -- from cityname0 (actual: cityname0)
                    CITYalias2 . CITY_NAME = \" city_name0 \" AND
                    -- to cityname1 (actual: cheapest)
                    CITYalias3 . CITY_CODE = AIRPORT_SERVICEalias3 . CITY_CODE AND 
                    -- to cityname1 (actual: cityname1)
                    CITYalias3 . CITY_NAME = \" city_name1 \" AND
                    -- from cityname0 (actual: cheapest)
                    FLIGHTalias1 . FROM_AIRPORT = AIRPORT_SERVICEalias2 . AIRPORT_CODE AND 
                    -- to cityname1 (actual: cheapest)
                    FLIGHTalias1 . TO_AIRPORT = AIRPORT_SERVICEalias3 . AIRPORT_CODE 
                ) AND 
                -- 1 way (actual: 1)
                FAREalias1 . ROUND_TRIP_REQUIRED = \" round_trip_required0 \" AND 
                -- null alignment (actual: cheapest)
                FLIGHT_FAREalias1 . FARE_ID = FAREalias1 . FARE_ID AND
                -- null alignment (actual: cheapest) 
                FLIGHTalias1 . FLIGHT_ID = FLIGHT_FAREalias1 . FLIGHT_ID AND 
                -- nonstop
                FLIGHTalias1 . STOPS = stops0 
        ) AND 
            -- 1 way (actual: 1)
            FAREalias0 . ROUND_TRIP_REQUIRED = \" round_trip_required0 \" AND 
            -- null alignment (actual: 1)
            FLIGHT_FAREalias0 . FARE_ID = FAREalias0 . FARE_ID AND 
            FLIGHTalias0 . FLIGHT_ID = FLIGHT_FAREalias0 . FLIGHT_ID ;
"""

# GROUP BY T1.id, T3.idx
    # HAVING T1.id > 100
    # ORDER BY school DESC|DESC
    # LIMIT 10

    import sqlparse
    parse_result = sqlparse.parse(sql)
    print(parse_result)


def main():
    dataset_root = Path('/path/to/improving-compgen-in-semparse/data/sql_data/')
    dataset_path = dataset_root / 'atis' / 'schema_full_split'  # or `new_question_split` for i.i.d. results

    generate_dataset(
        dataset_path / 'aligned_train.json',
        dataset_path / 'aligned_train.parse_comp_heuristics_false_analyze_nested_false.consecutive_utt.jsonl',
        compute_alignment=True,
        alignment_file_prefix='a0a017e1_rulerpt0'
    )

    generate_dataset(
        dataset_path / 'aligned_final_dev.json',
        dataset_path / 'aligned_final_dev.jsonl',
        compute_alignment=False
    )

    generate_dataset(
        dataset_path / 'final_test.json',
        dataset_path / 'final_test.jsonl',
        compute_alignment=False
    )


if __name__ == '__main__':
    main()

