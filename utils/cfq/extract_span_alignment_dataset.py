"""
Usage:
    preprocess SPLIT_ROOT [options]

Options:
    --dataset=<path>      Path to the original json dataset [default: None]
    --debug               Debug flag
"""

import json
import math
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any, Callable

from allennlp.data import Token
from allennlp.data.tokenizers import SpacyTokenizer
from docopt import docopt
from tqdm import tqdm

from utils.calflow.extract_span_alignment_dataset import (
    get_giza_alignments,
    SpanAlignment,
    MergedSpanAlignment,
    construct_alignment
)
import utils.cfq.cfq_utils.preprocess as cfq_preprocess


SPARQL_STMT_RE = re.compile(r'(m_\w*|M\d+|\?x\d+) (\S+) (m_\w*|M\d+|\?x\d+|\w+\.\w+)( \.)?')


def is_variable(text: str):
    return text.startswith('m_') or text.startswith('?') or text.startswith('M')


@dataclass
class Triple:
    __slots__ = ['token_span', 'tokens']

    tokens: List[Token]
    token_span: Tuple[int, int]

    def __repr__(self):
        return ' '.join([t.text for t in self.tokens])

    @property
    def variables(self) -> Tuple:
        var_1 = self.tokens[0]
        var_2 = self.tokens[-1] if self.tokens[-1].text != '.' else self.tokens[-2]

        return (var_1, var_2)

    @property
    def relation_span(self) -> Tuple[int, int]:
        return (self.op.text_id, self.op.text_id + 1)

    @property
    def op(self):
        return self.tokens[1]


@dataclass
class SparqlQuery:
    string: str
    triples: List[Triple]
    tokens: Optional[List[Token]]


tokenizer = SpacyTokenizer(split_on_spaces=True)


def parse_sparql(query_string: str) -> SparqlQuery:
    query_string = query_string.strip()
    query_tokens = tokenizer.tokenize(query_string)

    for token_idx, token in enumerate(query_tokens):
        token.text_id = token_idx

    triples = []

    matches = SPARQL_STMT_RE.finditer(query_string)
    for match in matches:
        stmt_start_char_idx = match.start()
        stmt_end_char_idx = match.end()

        stmt_start_token_idx = [idx for idx, token in enumerate(query_tokens) if token.idx == stmt_start_char_idx][0]
        stmt_end_token_idx = [idx for idx, token in enumerate(query_tokens) if token.idx_end == stmt_end_char_idx][0]

        last_token = query_tokens[stmt_end_token_idx]

        stmt_tokens = query_tokens[stmt_start_token_idx: stmt_end_token_idx + 1]
        triple = Triple(stmt_tokens, (stmt_start_token_idx, stmt_end_token_idx + 1))

        triples.append(triple)

    query = SparqlQuery(query_string, triples, query_tokens)

    return query


def find_aligned_source_span(
    target_span_start_idx, target_span_end_idx,
    span_alignments: List[SpanAlignment],
    existing_alignment_results: List[Dict],
    allow_source_overlap: bool = False,
    max_disconnected_alignment_distance: float = math.sqrt(2.0),
    target_tokens=None
) -> SpanAlignment:
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

    if not overlapped_alignments:
        return None

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

    # get rid of outliners
    has_removed_alignment = True
    while has_removed_alignment:
        has_removed_alignment = False

        for idx in range(len(overlapped_alignments) - 1):
            alignment = overlapped_alignments[idx]
            next_alignment = overlapped_alignments[idx + 1]

            dist = MergedSpanAlignment.get_minimal_child_span_distance(alignment, next_alignment)
            if dist > max_disconnected_alignment_distance:
                # keep the larger one
                if alignment.source_span[1] - alignment.source_span[0] > next_alignment.source_span[1] - next_alignment.source_span[0]:
                    idx_to_rm = idx + 1
                elif alignment.source_span[1] - alignment.source_span[0] < next_alignment.source_span[1] - next_alignment.source_span[0]:
                    idx_to_rm = idx
                else:
                    # remove the variable name first...
                    if alignment.target_span[1] - alignment.target_span[0] == 1 and is_variable(target_tokens[alignment.target_span[0]].text):
                        idx_to_rm = idx
                    else:
                        idx_to_rm = idx + 1

                del overlapped_alignments[idx_to_rm]
                has_removed_alignment = True
                break

    # # find the centroid cluster
    # center: Union[MergedSpanAlignment, SpanAlignment] = sorted(overlapped_alignments, key=lambda a: a.source_span[1] - a.source_span[0], reverse=True)[0]
    # for idx, alignment in enumerate(overlapped_alignments):
    #     if alignment != center:
    #         dist = MergedSpanAlignment.get_minimal_child_span_distance(alignment, center)
    #         if dist > max_disconnected_alignment_distance:
    #             span

    merged_alignment = MergedSpanAlignment(overlapped_alignments)

    if not allow_source_overlap and any(
        merged_alignment.source_overlap(m['alignment'])
        for m in existing_alignment_results
        if m['alignment'] is not None
    ):
        return None

    return merged_alignment


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


def find_aligned_source_spans(
    target_span_start_idx, target_span_end_idx,
    span_alignments: List[SpanAlignment],
    existing_alignment_results: List[Dict],
    source_tokens: List[str],
    target_triple: Triple,
    target_tokens: List[Token],
    allow_source_overlap: bool = False,
    max_disconnected_alignment_distance: float = math.sqrt(2.0),
) -> List[SpanAlignment]:
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

    if not overlapped_alignments:
        return None

    # special case only `executive` is aligned without `produced`
    if 'film.film.executive_produced_by' in ' '.join(token.text for token in target_triple.tokens):
        missing_relation_alignments = []
        for a_idx, alignment in enumerate(overlapped_alignments):
            rel_aligned_src_tokens = source_tokens[alignment.source_span[0]: alignment.source_span[1]]
            if (
                'executive' in rel_aligned_src_tokens and
                'produced' not in rel_aligned_src_tokens and
                alignment.source_span[1] < len(source_tokens) and
                source_tokens[alignment.source_span[1]] == 'produced' and
                not any(
                    alignment.source_span[1] == other_alignment.source_span[0]
                    for other_alignment
                    in overlapped_alignments
                )
            ):
                missing_relation_alignments.append(
                    # add alignment to the word `produced`
                    SpanAlignment(
                        (alignment.source_span[1], alignment.source_span[1] + 1),
                        alignment.target_span
                    )
                )
        if missing_relation_alignments:
            overlapped_alignments.extend(missing_relation_alignments)
            del missing_relation_alignments

    # expand the left boundary, incorporating more stop words
    allowable_relation_phrase_head_stopwords = ['a', 'was', 'is', 'were', 'the']
    allowable_relation_phrase_tail_stopwords = ['by', 'to', 'for', 'of', "'", 's']
    allowable_relation_phrase_mid_stopwords = ["'", "s", 'of', 'was', 'a', 'the']

    def is_stopword(__token: str):
        return __token in (
            allowable_relation_phrase_head_stopwords +
            allowable_relation_phrase_tail_stopwords +
            allowable_relation_phrase_mid_stopwords
        )

    overlapped_alignments.sort(key=lambda span: span.source_span)
    # including middle stopwords
    for idx in range(overlapped_alignments[0].source_span[0], overlapped_alignments[-1].source_span[1]):
        src_token = source_tokens[idx]
        if (
            not any(
                alignment.source_overlap(
                    SpanAlignment((idx, idx + 1), (-1, -1))
                )
                for alignment
                in overlapped_alignments
            ) and
            src_token in allowable_relation_phrase_mid_stopwords
        ):
            overlapped_alignments.append(
                SpanAlignment(
                    (idx, idx + 1),
                    (target_triple.tokens[0].text_id, target_triple.tokens[0].text_id + 1)
                )
            )

    overlapped_alignments.sort(key=lambda span: span.source_span)
    # add head stopwords
    idx = overlapped_alignments[0].source_span[0] - 1
    while idx >= 0 and source_tokens[idx] in allowable_relation_phrase_head_stopwords:
        overlapped_alignments.append(
            SpanAlignment(
                (idx, idx + 1),
                (target_triple.tokens[0].text_id, target_triple.tokens[0].text_id + 1)
            )
        )
        idx -= 1

    overlapped_alignments.sort(key=lambda span: span.source_span)
    idx = overlapped_alignments[-1].source_span[1]
    while idx < len(source_tokens) and source_tokens[idx] in allowable_relation_phrase_tail_stopwords:
        overlapped_alignments.append(
            SpanAlignment(
                (idx, idx + 1),
                (target_triple.tokens[-1].text_id, target_triple.tokens[-1].text_id + 1)
            )
        )
        idx += 1

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

    # I am going to add more heuristics!

    # get rid of outliners
    # has_removed_alignment = True
    # while has_removed_alignment:
    #     has_removed_alignment = False
    #
    #     for idx in range(len(overlapped_alignments) - 1):
    #         alignment = overlapped_alignments[idx]
    #         next_alignment = overlapped_alignments[idx + 1]
    #
    #         dist = MergedSpanAlignment.get_minimal_child_span_distance(alignment, next_alignment)
    #         if dist > max_disconnected_alignment_distance:
    #             # keep the larger one
    #             if alignment.source_span[1] - alignment.source_span[0] > next_alignment.source_span[1] - next_alignment.source_span[0]:
    #                 idx_to_rm = idx + 1
    #             elif alignment.source_span[1] - alignment.source_span[0] < next_alignment.source_span[1] - next_alignment.source_span[0]:
    #                 idx_to_rm = idx
    #             else:
    #                 # # remove the variable name first...
    #                 # if alignment.target_span[1] - alignment.target_span[0] == 1 and is_variable(
    #                 #     target_tokens[alignment.target_span[0]].text
    #                 # ):
    #                 #     idx_to_rm = idx
    #                 # else:
    #                 #     idx_to_rm = idx + 1
    #
    #             del overlapped_alignments[idx_to_rm]
    #             has_removed_alignment = True
    #             break

    # merged_alignment = MergedSpanAlignment(overlapped_alignments)
    # merged_alignments = [merged_alignment]
    alignments = overlapped_alignments

    variables = target_triple.variables
    for var in variables:
        for src_token_idx, src_token in enumerate(source_tokens):
            if src_token == var.text and not any(
                alignment.source_overlap(
                    SpanAlignment(
                        (src_token_idx, src_token_idx + 1),
                        (-1, -1)
                    )
                ) for alignment in alignments
            ):
                var_alignment = SpanAlignment((src_token_idx, src_token_idx + 1), (var.text_id, var.text_id + 1))
                alignments.append(var_alignment)

    # fill the gaps if the gap is composed of stop words
    alignments.sort(key=lambda a: a.source_span)
    for idx in range(len(overlapped_alignments) - 1):
        alignment = overlapped_alignments[idx]
        next_alignment = overlapped_alignments[idx + 1]

        all_stopwords = all(
            is_stopword(source_tokens[src_idx])
            for src_idx
            in range(alignment.source_span[1], next_alignment.source_span[0])
        )

        if all_stopwords:
            overlapped_alignments.append(
                SpanAlignment(
                    (alignment.source_span[1], next_alignment.source_span[0]),
                    (target_triple.tokens[0].text_id, target_triple.tokens[0].text_id + 1)
                )
            )

    for alignment in alignments:
        if source_tokens[alignment.source_span[0]] == 's':
            if alignment.source_span[1] - alignment.source_span[0] > 1:
                alignment.source_span = (alignment.source_span[0] + 1, alignment.source_span[1])

    alignments = merge_source_span_alignments(alignments)

    # remove dangeling stopwords
    cleaned_alignments = []
    for idx, alignment in enumerate(alignments):
        prev_alignment = None if idx == 0 else alignments[idx - 1]
        next_alignment = None if idx == len(alignments) - 1 else alignments[idx + 1]

        aligned_source_tokens = source_tokens[alignment.source_span[0]: alignment.source_span[1]]
        if all(
            is_stopword(token)
            for token
            in aligned_source_tokens
        ):
            pass
            # left = '' if not prev_alignment else source_tokens[prev_alignment.source_span[0]: prev_alignment.source_span[1]]
            # right = '' if not next_alignment else source_tokens[next_alignment.source_span[0]: next_alignment.source_span[1]]
            # print(f'Removed {left} **{aligned_source_tokens}** {right}')
        else:
            cleaned_alignments.append(alignment)

    if not allow_source_overlap and any(
        merged_alignment.source_overlap(m['alignment'])
        for m in existing_alignment_results
        if m['alignment'] is not None
    ):
        return None

    return cleaned_alignments


def extract_span_level_alignment(example: Dict, alignment_info: List, debug: bool = False):
    sparql = example['target']
    source_tokens = example['source'].split(' ')
    parsed_query = parse_sparql(sparql)

    if debug:
        print('*' * 20)
        print(example['example_idx'])
        print(example['source'])
        print(example['target'])
        for triple in parsed_query.triples:
            print('Triple: ' + repr(triple))
        print()

    alignment_tuples = []
    for src_idx, src_tok_align in enumerate(alignment_info):
        if src_tok_align:
            if not isinstance(src_tok_align, list):
                src_tok_align = [src_tok_align]

            for tgt_idx in src_tok_align:
                alignment_tuples.append((src_idx, tgt_idx))

    candidate_span_alignments = construct_alignment(None, None, alignment_tuples, merge=False)

    src_tgt_alignments = []
    if debug:
        print('Alignments: ')
    for triple in parsed_query.triples:
        stmt_start, stmt_end = triple.token_span
        # do not align ending punctuation
        if triple.tokens[-1].text == '.':
            stmt_end -= 1

        alignment = find_aligned_source_span(
            stmt_start, stmt_end,
            candidate_span_alignments,
            src_tgt_alignments,
            allow_source_overlap=True,
            target_tokens=parsed_query.tokens
        )

        if alignment:
            entry = {
                'triple': triple,
                'alignment': alignment
            }
            src_tgt_alignments.append(entry)

            if debug:
                print(f'{triple} <---> {" ".join(source_tokens[alignment.source_span[0]: alignment.source_span[1]])}')

    return src_tgt_alignments


def filter_rel_source_tokens(_tokens: List[str]) -> List[str]:
    return [
        tok
        for tok in _tokens
        if not is_variable(tok) and not is_stopword(tok) and tok not in string.punctuation
    ]


def extract_span_level_alignments_vanilla_representation(
    example: Dict,
    alignment_info_list: List,
    alignment_metadata: Dict,
    rule_set: List[str] = None,
    debug: bool = False
):
    assert not rule_set, 'This function does not use rule set!'

    sparql = example['target']
    source_tokens = example['source'].split(' ')
    parsed_query = parse_sparql(sparql)

    if debug:
        print('*' * 20)
        print(example['example_idx'])
        print(example['source'])
        print(example['target'])
        for triple in parsed_query.triples:
            print('Triple: ' + repr(triple))
        print()

    alignment_tuples = []
    for alignment_info in alignment_info_list:
        for src_idx, src_tok_align in enumerate(alignment_info):
            if src_tok_align:
                if not isinstance(src_tok_align, list):
                    src_tok_align = [src_tok_align]

                for tgt_idx in src_tok_align:
                    tgt_token = parsed_query.tokens[tgt_idx].text
                    src_token = source_tokens[src_idx]

                    is_valid = True
                    if tgt_token in {'{', '}', ',', '.', '?x0', '?x1', '?x2', '?x3', '?x4', 'COUNT', 'SELECT',
                                     'DISTINCT'}:
                        is_valid = False
                    elif src_token in {',', 'and', '.', ';', 'did', 'does', 'do'}:
                        is_valid = False
                    # variables should have the same alignments.
                    elif tgt_token.startswith('M') and src_token != tgt_token:
                        is_valid = False

                    if is_valid and (src_idx, tgt_idx) not in alignment_info:
                        alignment_tuples.append((src_idx, tgt_idx))

    # alignment_tuples = []
    # for src_idx, src_tok_align in enumerate(alignment_info):
    #     if src_tok_align:
    #         if not isinstance(src_tok_align, list):
    #             src_tok_align = [src_tok_align]
    #
    #         for tgt_idx in src_tok_align:
    #             alignment_tuples.append((src_idx, tgt_idx))

    candidate_span_alignments = construct_alignment(None, None, alignment_tuples, merge=False)

    # get relation alignments for each triple
    all_triple_relation_alignments = []
    for triple_idx, triple in enumerate(parsed_query.triples):
        relation_span: Tuple[int, int] = triple.relation_span
        triple_relation_alignments = []

        if triple.op.text != '!=':
            for idx, alignment in enumerate(candidate_span_alignments):
                aligned_src_tokens = source_tokens[alignment.source_span[0]: alignment.source_span[1]]

                if relation_span == alignment.target_span:
                    rel_tokens = filter_rel_source_tokens(aligned_src_tokens)
                    if rel_tokens:
                        triple_relation_alignments.append({
                            'idx': idx,
                            'tokens': rel_tokens,
                            'alignment': alignment,
                        })

        all_triple_relation_alignments.append(triple_relation_alignments)

    all_idx_to_del = []
    for triple_idx, triple in enumerate(parsed_query.triples):
        triple_relation_alignments = all_triple_relation_alignments[triple_idx]

        if len(triple_relation_alignments) > 1:
            idx_to_del = []
            for idx, alignment_entry in enumerate(triple_relation_alignments):
                alignment: SpanAlignment = alignment_entry['alignment']
                if any(
                    alignment.source_overlap(other_alignment['alignment'])
                    for other_idx, other_alignments in enumerate(all_triple_relation_alignments)
                    if other_idx != triple_idx
                    for other_alignment in other_alignments
                ):
                    idx_to_del.append(alignment_entry['idx'])

            if len(idx_to_del) == 1:
                all_idx_to_del.extend(idx_to_del)

    candidate_span_alignments = [
        alignment
        for idx, alignment in enumerate(candidate_span_alignments)
        if idx not in all_idx_to_del
    ]

    all_triple_alignments = []
    if debug:
        print('Alignments: ')
    for triple in parsed_query.triples:
        if triple.op.text == '!=':
            continue

        stmt_start, stmt_end = triple.token_span
        # do not align ending punctuation
        if triple.tokens[-1].text == '.':
            stmt_end -= 1

        alignments = find_aligned_source_spans(
            stmt_start, stmt_end,
            candidate_span_alignments,
            all_triple_alignments,
            allow_source_overlap=True,
            source_tokens=source_tokens,
            target_triple=triple,
            target_tokens=parsed_query.tokens
        )

        entry = {
            'triple': triple,
            'alignments': alignments
        }
        all_triple_alignments.append(entry)

        if alignments:
            for alignment in alignments:
                if debug:
                    print(f'{triple} <---> {" ".join(source_tokens[alignment.source_span[0]: alignment.source_span[1]])}')

    # for triple_idx, (triple, entry) in enumerate(zip(parsed_query.triples, all_triple_alignments)):
    #     triple_alignments = entry.get('alignments', [])
    #     if triple_alignments:
    #         relation_alignments = []
    #         for idx, alignment in enumerate(triple_alignments):
    #             aligned_source_tokens = source_tokens[alignment.source_span[0]: alignment.source_span[1]]
    #             rel_src_tokens = filter_rel_source_tokens(aligned_source_tokens)
    #             if rel_src_tokens:
    #                 relation_alignments.append({
    #                     'tokens': rel_src_tokens,
    #                     'alignment': alignment,
    #                     'idx': idx
    #                 })
    #
    #         if len(relation_alignments) > 1:
    #             idx_to_del = None
    #             for idx, alignment_entry in enumerate(relation_alignments):
    #                 alignment: SpanAlignment = alignment_entry['alignment']
    #                 if any(
    #                     alignment.source_overlap(other_alignment)
    #                     for other_idx, other_alignments in enumerate(all_triple_alignments)
    #                     if other_idx != triple_idx and other_alignments['alignments']
    #                     for other_alignment in other_alignments['alignments']
    #                 ):
    #                     idx_to_del = alignment_entry['idx']
    #
    #             if idx_to_del is not None:
    #                 del triple_alignments[idx_to_del]

    span_alignments = []
    for entry in all_triple_alignments:
        triple_alignments = entry.get('alignments')
        if triple_alignments:
            for alignment in triple_alignments:
                align_entry = {
                    'target_tokens_idx': tuple(entry['triple'].token_span),
                    'target_tokens': [t.text for t in entry['triple'].tokens],
                    'source_tokens': source_tokens[alignment.source_span[0]: alignment.source_span[1]],
                    'source_tokens_idx': tuple(alignment.source_span)
                }
                span_alignments.append(align_entry)

    token_level_alignments = []
    for src_idx, tgt_idx in alignment_tuples:
        token_level_alignments.append({
            'source_token_idx': src_idx,
            'target_token_idx': tgt_idx
        })

    if not span_alignments:
        print(example['idx'])
        raise RuntimeError('No alignment found.')

    return span_alignments, token_level_alignments


def include_variable_literal_alignment(
    var_name: str,
    token_span: Tuple[int, int],
    source_tokens: List[str],
    existing_alignments: List[SpanAlignment]
):
    for src_token_idx, src_token in enumerate(source_tokens):
        if src_token == var_name and not any(
            alignment.source_overlap(
                SpanAlignment(
                    (src_token_idx, src_token_idx + 1),
                    (-1, -1)
                )
            ) for alignment in existing_alignments
        ):
            var_alignment = SpanAlignment(
                (src_token_idx, src_token_idx + 1),
                token_span
            )
            existing_alignments.append(var_alignment)

    return existing_alignments


# expand the left boundary, incorporating more stop words
allowable_relation_phrase_head_stopwords = ['a', 'was', 'is', 'were', 'the', 'and']
allowable_relation_phrase_tail_stopwords = ['by', 'to', 'for', 'of', "'", 's']
allowable_relation_phrase_mid_stopwords = ["'", "s", 'of', 'was', 'a', 'the']
allowable_object_glue_stopwords = ['and', ',']


def is_stopword(token: str):
    return token in (
        allowable_relation_phrase_head_stopwords +
        allowable_relation_phrase_tail_stopwords +
        allowable_relation_phrase_mid_stopwords +
        allowable_object_glue_stopwords
    )


def find_overlapping_alignments(
    target_token_span: Tuple[int, int],
    input_span_alignments: List[SpanAlignment],
    source_tokens: List[str],
    allow_source_overlap: bool = False,
    align_to_stopwords: bool = True
) -> List[SpanAlignment]:
    target_span_start_idx, target_span_end_idx = target_token_span
    overlapped_alignments = [
        alignment
        for alignment in input_span_alignments
        if (
            (
                target_span_start_idx <= alignment.target_span[0] <= target_span_end_idx - 1 or
                target_span_start_idx <= alignment.target_span[1] - 1 <= target_span_end_idx - 1 or
                alignment.target_span[0] <= target_span_start_idx <= target_span_end_idx - 1 <= alignment.target_span[1] - 1
            ) and (
                allow_source_overlap or
                not any(
                    alignment.source_overlap(m['alignment'])
                    for m in input_span_alignments
                    if m['alignment'] is not None
                )
            ) and (
                align_to_stopwords or
                all(
                    is_stopword(token)
                    for token
                    in source_tokens[alignment.source_span[0]: alignment.source_span[1]]
                )
            )
        )
    ]

    return overlapped_alignments


def get_aligned_source_segments(alignments: List[SpanAlignment]) -> List[Tuple[int, int]]:
    if not alignments:
        return []

    aligned_source_segments = []
    cur_rel_segment = alignments[0].source_span
    for alignment in alignments[1:]:
        if alignment.source_span[0] == cur_rel_segment[1]:
            cur_rel_segment = (cur_rel_segment[0], alignment.source_span[1])
        elif alignment.source_span[0] > cur_rel_segment[1]:
            aligned_source_segments.append(cur_rel_segment)
            cur_rel_segment = alignment.source_span
        else:
            raise ValueError('unsorted relation alignments!')

    if cur_rel_segment not in aligned_source_segments:
        aligned_source_segments.append(cur_rel_segment)

    return aligned_source_segments


# noinspection DuplicatedCode
def find_aligned_source_spans_for_assertion(
    subject: List[Token],
    subject_token_span: Tuple[int, int],
    relation: List[Token],
    relation_token_span: Tuple[int, int],
    objects: List[Dict],
    objects_tokens_span: Tuple[int, int],
    input_span_alignments: List[SpanAlignment],
    existing_alignment_results: List[Dict],
    source_tokens: List[str],
    target_tokens: List[Token],
    allow_source_overlap: bool = False,
    max_disconnected_alignment_distance: float = math.sqrt(2.0),
    rule_set: List[str] = None
) -> List[SpanAlignment]:
    if rule_set is None:
        rule_set = [
            'include_middle_relation_stopwords',
            'add_missing_relation_alignment',
            'fix_relation_after_object'
        ]

    # find overlapping span alignments
    subject_candidate_alignments: List[SpanAlignment] = find_overlapping_alignments(
        subject_token_span, input_span_alignments,
        source_tokens,
        allow_source_overlap=allow_source_overlap,
        align_to_stopwords=False
    )

    for token in subject:
        if token.text in [f'M{i}' for i in range(10)]:
            include_variable_literal_alignment(
                token.text,
                subject_token_span,
                source_tokens,
                subject_candidate_alignments
            )

    relation_candidate_alignments: List[SpanAlignment] = find_overlapping_alignments(
        relation_token_span, input_span_alignments,
        source_tokens,
        allow_source_overlap=allow_source_overlap
    )

    # special case only `executive` is aligned without `produced`
    if 'film.film.executive_produced_by' in ' '.join(token.text for token in relation):
        missing_relation_alignments = []
        for a_idx, rel_alignment in enumerate(relation_candidate_alignments):
            rel_aligned_src_tokens = source_tokens[rel_alignment.source_span[0]: rel_alignment.source_span[1]]
            if (
                'executive' in rel_aligned_src_tokens and
                'produced' not in rel_aligned_src_tokens and
                rel_alignment.source_span[1] < len(source_tokens) and
                source_tokens[rel_alignment.source_span[1]] == 'produced' and
                not any(
                    rel_alignment.source_span[1] == other_alignment.source_span[0]
                    for other_alignment
                    in relation_candidate_alignments
                )
            ):
                missing_relation_alignments.append(
                    # add alignment to the word `produced`
                    SpanAlignment(
                        (rel_alignment.source_span[1], rel_alignment.source_span[1] + 1),
                        relation_token_span
                    )
                )
        if missing_relation_alignments:
            relation_candidate_alignments.extend(missing_relation_alignments)
            del missing_relation_alignments

    objects_candidate_alignments: List[SpanAlignment] = find_overlapping_alignments(
        objects_tokens_span, input_span_alignments,
        source_tokens,
        allow_source_overlap=allow_source_overlap
    )

    # for objects, add variable literal alignments
    for obj in objects:
        if obj['object'] in [f'M{i}' for i in range(10)]:
            include_variable_literal_alignment(
                obj['object'],
                obj['token_span'],
                source_tokens,
                objects_candidate_alignments
            )

    objects_candidate_alignments.sort(key=lambda a: a.source_span)

    # fill the gaps in objects if the gap is composed of stop words
    for idx in range(len(objects_candidate_alignments) - 1):
        alignment = objects_candidate_alignments[idx]
        next_alignment = objects_candidate_alignments[idx + 1]

        all_stopwords = all(
            is_stopword(source_tokens[src_idx])
            for src_idx
            in range(alignment.source_span[1], next_alignment.source_span[0])
        )

        if all_stopwords:
            objects_candidate_alignments.append(
                SpanAlignment(
                    (alignment.source_span[1], next_alignment.source_span[0]),
                    objects_tokens_span
                )
            )

    if not subject_candidate_alignments and not relation_candidate_alignments and not objects_candidate_alignments:
        raise ValueError('No alignments!')
        return []  # noqa

    subject_candidate_alignments.sort(key=lambda a: a.source_span)
    relation_candidate_alignments.sort(key=lambda a: a.source_span)
    objects_candidate_alignments.sort(key=lambda a: a.source_span)

    objects_aligned_source_segments = get_aligned_source_segments(objects_candidate_alignments)

    if 'include_middle_relation_stopwords' in rule_set:
        # including relation middle stopwords
        if relation_candidate_alignments:
            for idx in range(
                relation_candidate_alignments[0].source_span[0],
                relation_candidate_alignments[-1].source_span[1]
            ):
                src_token = source_tokens[idx]
                if (
                    not any(
                        alignment.source_overlap(
                            SpanAlignment((idx, idx + 1), (-1, -1))
                        )
                        for alignment
                        in relation_candidate_alignments
                    ) and
                    src_token in allowable_relation_phrase_mid_stopwords
                ):
                    relation_candidate_alignments.append(
                        SpanAlignment(
                            (idx, idx + 1),
                            relation_token_span
                        )
                    )

    relation_candidate_alignments.sort(key=lambda a: a.source_span)

    # for relations, we need to identify consecutive relations within those alignments
    # right now, we do it in a naviave way and try to append stopwords for each relations
    relation_aligned_source_segments = get_aligned_source_segments(relation_candidate_alignments)

    if 'add_missing_relation_alignment' in rule_set:
        # find the object which does not have a leading relation mention
        # if len(relation_aligned_source_segments) < len(objects_aligned_source_segments):
        missing_relation_candidate_alignments = []
        if len(objects) > 1:
            for rel_alignment in relation_candidate_alignments:
                rel_alignment_src_tokens = source_tokens[rel_alignment.source_span[0]: rel_alignment.source_span[1]]

                if all(is_stopword(token) for token in rel_alignment_src_tokens):
                    continue

                for src_idx in range(len(source_tokens)):
                    proposal_aligned_src_span = (src_idx, src_idx + rel_alignment.source_span[1] - rel_alignment.source_span[0])
                    if proposal_aligned_src_span[1] > len(source_tokens):
                        continue

                    proposal_aligned_src_tokens = source_tokens[proposal_aligned_src_span[0]: proposal_aligned_src_span[1]]
                    if proposal_aligned_src_tokens == rel_alignment_src_tokens and not any(
                        proposal_aligned_src_span == a.source_span
                        for a
                        in relation_candidate_alignments + missing_relation_candidate_alignments
                    ):
                        missing_alignment = SpanAlignment(proposal_aligned_src_span, rel_alignment.target_span)
                        missing_relation_candidate_alignments.append(missing_alignment)

        if False and len(objects) > 1:
            for seg_idx, obj_src_segment in enumerate(objects_aligned_source_segments):
                prev_obj_src_seg_right = 0 if seg_idx == 0 else objects_aligned_source_segments[seg_idx - 1][1]

                has_leading_relation = any(
                    rel_src_seg[1] <= obj_src_segment[0] and rel_src_seg[0] > prev_obj_src_seg_right
                    for rel_src_seg
                    in relation_aligned_source_segments
                )

                if not has_leading_relation:
                    src_span_to_search = (prev_obj_src_seg_right, obj_src_segment[0])
                    if src_span_to_search[1] - src_span_to_search[0] > 0:
                        for alignment in relation_candidate_alignments:
                            aligned_src_tokens = source_tokens[alignment.source_span[0]: alignment.source_span[1]]
                            if all(is_stopword(token) for token in aligned_src_tokens):
                                continue

                            aligned_tgt_tokens = [
                                t
                                for t
                                in target_tokens[alignment.target_span[0]: alignment.target_span[1]]
                            ]

                            if relation == aligned_tgt_tokens:
                                for src_idx in range(*src_span_to_search):
                                    aligned_src_span = (src_idx, src_idx + alignment.source_span[1] - alignment.source_span[0])
                                    if source_tokens[aligned_src_span[0]: aligned_src_span[1]] == aligned_src_tokens:
                                        missing_alignment = SpanAlignment(aligned_src_span, alignment.target_span)
                                        missing_relation_candidate_alignments.append(missing_alignment)

        if missing_relation_candidate_alignments:
            relation_candidate_alignments.extend(missing_relation_candidate_alignments)
            relation_candidate_alignments = merge_source_span_alignments(relation_candidate_alignments)
            relation_candidate_alignments.sort(key=lambda a: a.source_span)
            relation_aligned_source_segments = get_aligned_source_segments(relation_candidate_alignments)

    # add head stopwords for each relation mention
    for idx, rel_src_segment in enumerate(relation_aligned_source_segments):
        if idx >= 1:
            prev_src_segment_right = relation_aligned_source_segments[idx - 1][1]
        else:
            prev_src_segment_right = 0

        # add head stopwords for relation
        # idx = relation_candidate_alignments[0].source_span[0] - 1
        idx = rel_src_segment[0] - 1
        while idx >= prev_src_segment_right and source_tokens[idx] in allowable_relation_phrase_head_stopwords:
            relation_candidate_alignments.append(
                SpanAlignment(
                    (idx, idx + 1),
                    relation_token_span
                )
            )
            idx -= 1

    relation_candidate_alignments.sort(key=lambda a: a.source_span)

    # add tail stopwords for each relation mention
    relation_aligned_source_segments = get_aligned_source_segments(relation_candidate_alignments)
    for idx, rel_src_segment in enumerate(relation_aligned_source_segments):
        if idx < len(relation_aligned_source_segments) - 1:
            next_src_segment_left = relation_aligned_source_segments[idx + 1][0]
        else:
            next_src_segment_left = len(source_tokens)

        idx = rel_src_segment[1]
        while idx < next_src_segment_left and source_tokens[idx] in allowable_relation_phrase_tail_stopwords:
            relation_candidate_alignments.append(
                SpanAlignment(
                    (idx, idx + 1),
                    relation_token_span
                )
            )
            idx += 1

    relation_candidate_alignments.sort(key=lambda a: a.source_span)

    # add head stopwords before object and a distant relation
    if objects_candidate_alignments:
        objects_aligned_source_segments = get_aligned_source_segments(objects_candidate_alignments)
        for src_segment in objects_aligned_source_segments:
            idx = src_segment[0] - 1
            left_idx = (
                relation_candidate_alignments[-1].source_span[1]
                if relation_candidate_alignments
                else 0
            )
            while idx >= left_idx and source_tokens[idx] in allowable_relation_phrase_tail_stopwords:
                relation_candidate_alignments.append(
                    SpanAlignment(
                        (idx, idx + 1),
                        relation_token_span
                    )
                )
                idx -= 1

    relation_candidate_alignments.sort(key=lambda a: a.source_span)

    if 'fix_relation_after_object' in rule_set:
        # [M3] 's [child]
        if relation_candidate_alignments and objects_candidate_alignments:
            start_idx = objects_candidate_alignments[-1].source_span[1]
            end_idx = relation_candidate_alignments[0].source_span[0]
            if start_idx < end_idx:
                for idx in range(start_idx, min(end_idx, start_idx + 2)):
                    if source_tokens[idx] in allowable_relation_phrase_mid_stopwords:
                        relation_candidate_alignments.append(
                            SpanAlignment(
                                (idx, idx + 1),
                                relation_token_span
                            )
                        )

    alignments = subject_candidate_alignments + relation_candidate_alignments + objects_candidate_alignments

    # fill the gaps of all alignments if the gap is composed of stop words
    alignments.sort(key=lambda a: a.source_span)
    for idx in range(len(alignments) - 1):
        alignment = alignments[idx]
        next_alignment = alignments[idx + 1]

        if alignment.source_span[1] >= next_alignment.source_span[0]:
            continue

        all_stopwords = all(
            is_stopword(source_tokens[src_idx])
            for src_idx
            in range(alignment.source_span[1], next_alignment.source_span[0])
        )

        if all_stopwords:
            alignments.append(
                SpanAlignment(
                    (alignment.source_span[1], next_alignment.source_span[0]),
                    (alignment.target_span[0], alignment.target_span[1])
                )
            )

    for alignment in alignments:
        if source_tokens[alignment.source_span[0]] == 's':
            if alignment.source_span[1] - alignment.source_span[0] > 1:
                alignment.source_span = (alignment.source_span[0] + 1, alignment.source_span[1])

    alignments = merge_source_span_alignments(alignments)

    if not allow_source_overlap and any(
        merged_alignment.source_overlap(m['alignment'])
        for m in existing_alignment_results
        if m['alignment'] is not None
    ):
        raise ValueError()
        return []  # noqa

    return alignments


def get_span_index_from_char_offsets(
    tokens: List[Token],
    tgt_span_char_offset: int,
    tgt_span_end_char_offset: int
) -> Tuple[int, int]:
    start_idx = [idx for idx, token in enumerate(tokens) if token.idx == tgt_span_char_offset][0]
    end_idx = [idx for idx, token in enumerate(tokens) if token.idx_end == tgt_span_end_char_offset][0] + 1

    return start_idx, end_idx


def parse_simplified_sparql(
    sparql: str,
    metadata: Dict
) -> Tuple[List[Token], Dict[str, Any]]:
    query_string = sparql.strip()
    query_tokens = tokenizer.tokenize(query_string)

    for token_idx, token in enumerate(query_tokens):
        token.text_id = token_idx

    for subject in metadata['subjects']:
        subject_token_span = get_span_index_from_char_offsets(query_tokens, *subject['char_offset'])
        subject['token_span'] = subject_token_span
        subject_tokens = query_tokens[subject_token_span[0]: subject_token_span[1]]
        subject['tokens'] = subject_tokens

        for relation in subject['relations']:
            relation_token_span = get_span_index_from_char_offsets(query_tokens, *relation['char_offset'])
            relation_tokens = query_tokens[relation_token_span[0]: relation_token_span[1]]
            relation['tokens'] = relation_tokens
            relation['token_span'] = relation_token_span

            objects = relation['objects']
            objects_tokens_span = get_span_index_from_char_offsets(query_tokens, objects[0]['char_offset'][0], objects[-1]['char_offset'][-1])
            relation['objects_tokens_span'] = objects_tokens_span

            for obj in objects:
                obj_token_span = get_span_index_from_char_offsets(query_tokens, *obj['char_offset'])
                obj['token_span'] = obj_token_span

    return query_tokens, metadata


def extract_span_level_alignments_grouped_representation(
    example: Dict,
    alignment_info_list: List,
    alignment_metadata: Dict,
    rule_set: List[str] = None,
    debug: bool = True
):
    sparql = example['target']
    source_tokens = example['source'].split(' ')
    parsed_query, sparql_metadata = parse_simplified_sparql(sparql, alignment_metadata)

    if debug:
        print('*' * 20)
        print(example['example_idx'])
        print(example['source'])
        print(example['target'])
        print()

    alignment_tuples = []
    for alignment_info in alignment_info_list:
        for src_idx, src_tok_align in enumerate(alignment_info):
            if src_tok_align:
                if not isinstance(src_tok_align, list):
                    src_tok_align = [src_tok_align]

                for tgt_idx in src_tok_align:
                    tgt_token = parsed_query[tgt_idx].text
                    src_token = source_tokens[src_idx]

                    is_valid = True
                    if tgt_token in {'{', '}', ',', '.', '?x0', '?x1', '?x2', '?x3', '?x4', 'COUNT', 'SELECT', 'DISTINCT'}:
                        is_valid = False
                    elif src_token in {',', 'and', '.', ';', 'did', 'does', 'do'}:
                        is_valid = False

                    if is_valid and (src_idx, tgt_idx) not in alignment_info:
                        alignment_tuples.append((src_idx, tgt_idx))

    alignment_tuples.sort(key=lambda x: x)  # sort first by src_idx and then by tgt_idx

    input_span_alignments = construct_alignment(None, None, alignment_tuples, merge=False)

    src_tgt_alignments = []
    if debug:
        print('Alignments: ')

    assertion_group: Dict
    for assertion_group in sparql_metadata['subjects']:
        subject = assertion_group['subject']
        subject_token_span: Tuple[int, int] = assertion_group['token_span']
        for rel_idx, relation in enumerate(assertion_group['relations']):
            if relation['relation'] == 'not':
                continue

            relation_token_span = relation['token_span']

            objects = relation['objects']
            objects_tokens_span = relation['objects_tokens_span']

            # get alignment of <subject, relation, objects> tuple
            alignments = find_aligned_source_spans_for_assertion(
                assertion_group['tokens'], subject_token_span,
                relation['tokens'], relation_token_span,
                objects, objects_tokens_span,
                input_span_alignments,
                src_tgt_alignments,
                rule_set=rule_set,
                allow_source_overlap=True,
                source_tokens=source_tokens,
                target_tokens=parsed_query
            )

            if alignments:
                for alignment in alignments:
                    entry = {
                        'subject': ' '.join(token.text for token in assertion_group['tokens']),
                        'relation': ' '.join(token.text for token in relation['tokens']),
                        'objects': [obj['object'] for obj in objects],
                        'alignment': alignment
                    }

                    aligned_source_tokens = source_tokens[alignment.source_span[0]: alignment.source_span[1]]

                    # subject {
                    subject_span = (subject_token_span[0], subject_token_span[1] + 1)
                    subj_alignment_entry = {
                        'target_tokens_idx': subject_span,
                        'target_tokens': [token.text for token in parsed_query[subject_span[0]: subject_span[1]]],
                        'source_tokens_idx': alignment.source_span,
                        'source_tokens': aligned_source_tokens
                    }
                    src_tgt_alignments.append(subj_alignment_entry)

                    rel_alignment_entry = {
                        'target_tokens_idx': relation_token_span,
                        'target_tokens': [token.text for token in parsed_query[relation_token_span[0]: relation_token_span[1]]],
                        'source_tokens_idx': alignment.source_span,
                        'source_tokens': aligned_source_tokens
                    }
                    src_tgt_alignments.append(rel_alignment_entry)

                    # { objects }
                    objects_span = (objects_tokens_span[0] - 1, objects_tokens_span[1] + 1)
                    last_rel = rel_idx == len(assertion_group['relations']) - 1
                    if last_rel:
                        objects_span = (objects_span[0], objects_span[1] + 1)

                    objects_alignment_entry = {
                        'target_tokens_idx': objects_span,
                        'target_tokens': [token.text for token in parsed_query[objects_span[0]: objects_span[1]]],
                        'source_tokens_idx': alignment.source_span,
                        'source_tokens': aligned_source_tokens
                    }
                    src_tgt_alignments.append(objects_alignment_entry)

                    if debug:
                        print(
                            f'{" ".join(subj_alignment_entry["target_tokens"])}'
                            f' {" ".join(rel_alignment_entry["target_tokens"])} '
                            f' {" ".join(objects_alignment_entry["target_tokens"])} <---> '
                            f'{" ".join(aligned_source_tokens)}'
                        )

            pass

    token_level_alignments = []
    for src_idx, tgt_idx in alignment_tuples:
        token_level_alignments.append({
            'source_token_idx': src_idx,
            'target_token_idx': tgt_idx
        })

    return src_tgt_alignments, token_level_alignments


def is_non_conjunctive_example(example):
    return not any(
        rule['stringValue'].startswith('CONJUNCT=')
        for rule
        in example['ruleIds']
    )


def get_tags(example):
    if is_non_conjunctive_example(example):
        return ['non_conjunctive']  # recursive examples
    else:
        return ['conjunctive']


def preprocess_cfq_split(
    dataset_root: Path,
    output_root: Path,
    label: str,
    dataset: Optional[List] = None,
    dataset_source_index: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    rule_set: List[str] = None
):
    dataset_root = dataset_root.expanduser()

    train_examples = preprocess_cfq(
        # dataset_root / 'train' / 'train_encode.top1000.txt',
        dataset_root / 'train' / 'train_encode.txt',
        dataset_root / 'train' / 'train_decode.txt',
        dataset_root / 'train' / 'train.debug.jsonl',
        dataset_root / 'train' / 's2t.VA3.final',
        dataset_root / 'train' / 't2s.VA3.final',
        alignment_dir='t2s',
        dataset=dataset,
        dataset_source_index=dataset_source_index,
        example_filters=[is_non_conjunctive_example],
    )

    dev_examples = preprocess_cfq(
        dataset_root / 'dev' / 'dev_encode.txt',
        dataset_root / 'dev' / 'dev_decode.txt',
        dataset_root / 'dev' / 'dev.jsonl',
        dataset=dataset,
        dataset_source_index=dataset_source_index,
        example_filters=[is_non_conjunctive_example]
    )

    test_examples = preprocess_cfq(
        dataset_root / 'test' / 'test_encode.txt',
        dataset_root / 'test' / 'test_decode.txt',
        dataset_root / 'test' / 'test.jsonl',
        dataset=dataset,
        dataset_source_index=dataset_source_index,
        example_filters=[is_non_conjunctive_example]
    )

    dump_dataset_for_t5_finetuning(train_examples, dev_examples, test_examples, dataset_root / 't5_finetune')


def preprocess_cfq_split_full_data(
    dataset_root: Path,
    output_root: Path,
    label: str,
    alignment_extraction_function: Callable,
    alignment_dir: str = 'both',
    dataset: Optional[List] = None,
    dataset_source_index: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    rule_set: List[str] = None
):
    dataset_root = dataset_root.expanduser()
    output_root = output_root.expanduser()

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'dev').mkdir(parents=True, exist_ok=True)
    (output_root / 'test').mkdir(parents=True, exist_ok=True)

    train_examples = preprocess_cfq(
        # dataset_root / 'train' / 'train_encode.top1000.txt',
        dataset_root / 'train' / 'train_encode.txt',
        dataset_root / 'train' / f'train_decode_{label}.txt',
        output_root / 'train' / 'train.jsonl',
        dataset_root / 'train' / f'{label}_s2t.VA3.final',
        dataset_root / 'train' / f'{label}_t2s.VA3.final',
        alignment_metadata_path=dataset_root / 'train' / f'train_decode.{label}.meta.jsonl',
        alignment_dir=alignment_dir,
        alignment_extraction_function=alignment_extraction_function,
        dataset=dataset,
        dataset_source_index=dataset_source_index,
        example_taggers=[get_tags],
        rule_set=rule_set,
        debug=debug
    )

    if not debug:
        dev_examples = preprocess_cfq(
            dataset_root / 'dev' / 'dev_encode.txt',
            dataset_root / 'dev' / f'dev_decode_{label}.txt',
            output_root / 'dev' / 'dev.jsonl',
            dataset=dataset,
            dataset_source_index=dataset_source_index,
            example_taggers=[get_tags],
            debug=debug
        )

        test_examples = preprocess_cfq(
            dataset_root / 'test' / 'test_encode.txt',
            dataset_root / 'test' / f'test_decode_{label}.txt',
            output_root / 'test' / 'test.jsonl',
            dataset=dataset,
            dataset_source_index=dataset_source_index,
            example_taggers=[get_tags],
            debug=debug
        )

    # dump_dataset_for_t5_finetuning(train_examples, dev_examples, test_examples, dataset_root / 't5_finetune')


def preprocess_cfq(
    src_file_path: Path,
    tgt_file_path: Path,
    output_path: Path,
    s2t_file_path: Optional[Path] = None,
    t2s_file_path: Optional[Path] = None,
    alignment_metadata_path: Optional[Path] = None,
    alignment_dir: Optional[str] = 't2s',
    dataset: Optional[List] = None,
    dataset_source_index: Optional[Dict[str, Any]] = None,
    example_taggers: List[Callable] = None,
    rule_set: List[str] = None,
    debug: bool = False,
    alignment_extraction_function: Callable = extract_span_level_alignments_grouped_representation
) -> List[Dict]:
    example_taggers = example_taggers or []

    alignments = None
    alignments_metadata = []
    if s2t_file_path:
        alignment_output_file_name = s2t_file_path.name.rpartition('_s2t.VA3.final')[0] + '.alignments.json'
        alignment_output_file = s2t_file_path.parent / alignment_output_file_name
        # if alignment_output_file.exists():
        #     alignments = json.load(alignment_output_file.open())
        # else:
        alignments = get_giza_alignments(
            s2t_file_path.expanduser(),
            t2s_file_path.expanduser(),
            alignment_output_file
        )

        if alignment_metadata_path:
            alignments_metadata = [
                json.loads(line)
                for line
                in alignment_metadata_path.open()
            ]

    examples = []
    for idx, (src_line, tgt_line) in tqdm(
        enumerate(
            zip(
                src_file_path.expanduser().open(),
                tgt_file_path.expanduser().open()
            )
        ),
        total=len(src_file_path.expanduser().open().readlines())
    ):
        source = src_line.strip()
        target = tgt_line.strip()
        tags = []

        # if not debug and example_taggers:
        if example_taggers:
            example = dataset[dataset_source_index[source]]
            for func in example_taggers:
                tags.extend(func(example))

            # if not any(func(example) for func in example_filters):
            #     continue

        if debug and ('non_conjunctive' not in tags or idx > 2000):
            continue

        example_entry = {
            'example_idx': idx,
            'source': source,
            'target': target,
            'tags': tags,
            'alignments': None,
        }

        if alignments:
            alignment_entry = alignments[idx]
            if alignment_dir in ['s2t', 't2s']:
                alignment_info = [alignment_entry[f'{alignment_dir}_alignment']]
            else:
                alignment_info = [alignment_entry[f's2t_alignment'], alignment_entry[f't2s_alignment']]

            alignment_metadata = alignments_metadata[idx]
            span_alignments, token_level_alignments = alignment_extraction_function(
                example_entry,
                alignment_info,
                alignment_metadata,
                rule_set=rule_set,
                debug=debug
            )
            # source_tokens = example_entry['source'].split(' ')

            # alignment_entries = []

            # for entry in span_alignments:
            #     # align_entry = {
            #     #     'target_tokens_idx': tuple(entry['triple'].token_span),
            #     #     'target_tokens': [t.text for t in entry['triple'].tokens],
            #     #     'source_tokens': source_tokens[entry['alignment'].source_span[0]: entry['alignment'].source_span[1]],
            #     #     'source_tokens_idx': tuple(entry['alignment'].source_span)
            #     # }
            #     # alignment_entries.append(align_entry)
            #     align_entry = {
            #         'target_tokens_idx': tuple(entry['target_'].token_span),
            #         'target_tokens': [t.text for t in entry['triple'].tokens],
            #         'source_tokens': source_tokens[entry['alignment'].source_span[0]: entry['alignment'].source_span[1]],
            #         'source_tokens_idx': tuple(entry['alignment'].source_span)
            #     }
            #     alignment_entries.append(align_entry)

            example_entry['alignments'] = span_alignments
            example_entry['token_level_alignments'] = token_level_alignments

            # token_level_alignments = extract_token_level_alignments(alignment_info, alignment_entry)
            #example_entry['token_level_alignments'] = token_level_alignments

        examples.append(example_entry)

    with output_path.expanduser().open('w') as f:
        for example_entry in examples:
            line = json.dumps(example_entry)
            f.write(line + '\n')

    return examples


def extract_token_level_alignments(alignment, alignment_entry):
    alignment_list = []
    for src_tok_idx, src_token in enumerate(alignment_entry['source_tokens']):
        tgt_tok_ids = alignment[src_tok_idx]

        if tgt_tok_ids is not None:
            if not isinstance(tgt_tok_ids, list):
                tgt_tok_ids = [tgt_tok_ids]

            for tgt_tok_id in tgt_tok_ids:
                entry = {
                    'source_token_idx': src_tok_idx,
                    'target_token_idx': tgt_tok_id
                }
                if entry not in alignment_list:
                    alignment_list.append(entry)

    return alignment_list


def dump_debug_dataset():
    dataset_file = Path('~/Research/datasets/cfq/dataset.json')
    dataset = json.load(dataset_file.expanduser().open())

    dataset_source_index = {
        cfq_preprocess.tokenize_punctuation(
            cfq_preprocess.tokenize_punctuation(example['questionPatternModEntities'])
        ): idx
        for idx, example
        in enumerate(dataset)
    }

    split_file = Path('~/Research/datasets/cfq/mcd2/train/train_encode.txt').expanduser()
    top_k = split_file.open().readlines()[:1000]
    sample_dataset = []
    with split_file.with_suffix('.top1000.txt').open('w') as f:
        for line in top_k:
            entry = dataset[dataset_source_index[line.strip()]]
            sample_dataset.append(entry)
            f.write(line)

    with split_file.with_name('train.top1000.dataset.json').open('w') as f:
        json.dump(sample_dataset, f)


def dump_dataset_for_t5_finetuning(
    train_examples: List[Dict],
    dev_examples: List[Dict],
    test_examples: List[Dict],
    output_folder_path: Path
):
    output_folder_path.mkdir(exist_ok=True, parents=True)
    for split in ['train', 'val', 'test']:
        with (output_folder_path / f'{split}.source').open('w') as f_src, (output_folder_path / f'{split}.target').open('w') as f_tgt:
            examples = {'train': train_examples, 'val': dev_examples, 'test': test_examples}[split]
            for example in examples:
                f_src.write(example['source'] + '\n')
                f_tgt.write(example['target'] + '\n')


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset_root = Path(args['SPLIT_ROOT'])
    label = 'simplified'

    rule_set = [
        'include_middle_relation_stopwords',
        'fix_relation_after_object',
        # 'add_missing_relation_alignment',
    ]
    # rule_set = None

    output_root = dataset_root.parent / f'{dataset_root.name}_{label}_mid_rel_sw_fix_rel_after_obj'
    assert not output_root.exists()
    print(f'output dataset to {output_root}')

    kwargs = {}

    if args['--dataset'] != 'None':
        print(f'Loading dataset @ {args["--dataset"]}')
        dataset = json.load(Path(args['--dataset']).open())
        dataset_source_index = {
            cfq_preprocess.tokenize_punctuation(
                cfq_preprocess.tokenize_punctuation(example['questionPatternModEntities'])
            ): idx
            for idx, example
            in enumerate(dataset)
        }

        kwargs = {
            'dataset': dataset,
            'dataset_source_index': dataset_source_index
        }

    preprocess_cfq_split_full_data(
        dataset_root,
        output_root,
        debug=args['--debug'],
        label=label,
        rule_set=rule_set,
        alignment_dir='both',
        alignment_extraction_function=extract_span_level_alignments_grouped_representation,
        # alignment_extraction_function=extract_span_level_alignments_vanilla_representation,
        **kwargs
    )
