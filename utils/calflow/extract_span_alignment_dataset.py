import itertools
import json
import math
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Set, TypeVar, Any, Dict, Union, Optional

from dataflow.core.linearize import seq_to_sexp, sexp_to_seq
from tqdm import tqdm

from dataflow.core.sexp import Sexp

from pynlpl.formats.giza import *

from utils.sexp_utils import find_sequence


def canonicalize_plan_tokens(plan_tokens: List[str]) -> (List[str], List[int]):
    canonical_tokens = []
    canonical_token_index_map = []
    for original_idx, token in enumerate(plan_tokens):
        if (
            token not in {'(', ')', '#', '"'}
            and not token.startswith("'?")
            # and not t.startswith(":")
        ):
            canonical_tokens.append(token)
            canonical_token_index_map.append(original_idx)

    return canonical_tokens, canonical_token_index_map


def canonicalize_plan(plan: str) -> str:
    return ' '.join(
        canonicalize_plan_tokens(plan.split(' '))[0]
    )


def get_giza_alignments(
    s2t_alignment_file: Path,
    t2s_alignment_file: Path,
    output: Path = None
):
    # '/giza-pp/GIZA++-v2/s2t_train_only.VA3.final'
    s2t_align_reader = MultiWordAlignment(str(s2t_alignment_file.expanduser()))
    # '/giza-pp/GIZA++-v2/t2s_train_only.VA3.final'
    t2s_align_reader = WordAlignment(str(t2s_alignment_file.expanduser()))

    bi_align_reader = IntersectionAlignment(
        str(s2t_alignment_file),
        str(t2s_alignment_file)
    )

    alignments = []
    for idx, (bi_align, s2t_align, t2s_align) in enumerate(zip(bi_align_reader, s2t_align_reader, t2s_align_reader)):
        src, tgt, alignment, is_valid = bi_align
        # print(bi_align)

        alignments.append({
            'example_idx': idx,
            'source_tokens': src,
            'target_tokens': tgt,
            'alignment': alignment,
            's2t_alignment': s2t_align[-1],
            't2s_alignment': t2s_align[-1],
            'is_valid_alignment': is_valid
        })

    if output:
        with output.open('w') as f:
            json.dump(alignments, f)

    return alignments


def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@dataclass
class SpanAlignment(object):
    source_span: Tuple[int, int]
    target_span: Tuple[int, int]

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            self.source_span[0] + (self.source_span[1] - self.source_span[0]) / 2,
            self.target_span[0] + (self.target_span[1] - self.target_span[0]) / 2
        )

    def distance(self, other: 'SpanAlignment') -> float:
        left = other.source_span[1] < self.source_span[0]
        right = self.source_span[1] < other.source_span[0]
        bottom = self.target_span[1] < other.target_span[0]
        top = other.target_span[1] < self.target_span[0]

        if top and left:
            return euclidean_dist((self.source_span[0], self.target_span[0]), (other.source_span[1], other.target_span[1]))
        elif left and bottom:
            return euclidean_dist((self.source_span[0], self.target_span[1]), (other.source_span[1], other.target_span[0]))
        elif bottom and right:
            return euclidean_dist((self.source_span[1], self.target_span[1]), (other.source_span[0], other.target_span[0]))
        elif right and top:
            return euclidean_dist((self.source_span[1], self.target_span[0]), (other.source_span[0], other.target_span[1]))
        elif left:
            return self.source_span[0] - other.source_span[1]
        elif right:
            return other.source_span[0] - self.source_span[1]
        elif bottom:
            return other.target_span[0] - self.target_span[1]
        elif top:
            return self.target_span[0] - other.target_span[1]
        else:
            return 0.

    def source_distance(self, other: 'SpanAlignment') -> float:
        left = other.source_span[1] < self.source_span[0]
        right = self.source_span[1] < other.source_span[0]
        bottom = self.target_span[1] < other.target_span[0]
        top = other.target_span[1] < self.target_span[0]

        if left:
            return math.fabs(self.source_span[0] - other.source_span[1])
        elif right:
            return math.fabs(self.source_span[1] - other.source_span[0])
        else:
            return 0.

    def overlaps(self, other: 'SpanAlignment') -> bool:
        raise Exception('Buggy!')
        return (
            (
                self.source_span[0] <= other.source_span[0] <= self.source_span[1] or
                self.source_span[0] <= other.source_span[1] <= self.source_span[1]
            ) and
            (
                self.target_span[0] <= other.target_span[0] <= self.target_span[1] or
                self.target_span[1] <= other.target_span[1] <= self.target_span[1]
            )
        )

    def source_overlap(self, other: 'SpanAlignment') -> bool:
        return (
            self.source_span[0] <= other.source_span[0] <= self.source_span[1] - 1 or
            self.source_span[0] <= other.source_span[1] - 1 <= self.source_span[1] - 1 or
            other.source_span[0] <= self.source_span[0] and self.source_span[1] <= other.source_span[1]
        )


@dataclass
class MergedSpanAlignment(SpanAlignment):
    spans: List[SpanAlignment]
    nested_spans: List[Union[SpanAlignment, 'MergedSpanAlignment']]

    def __init__(self, spans):
        flattened_spans = []
        nested_spans = []
        for span in spans:
            nested_spans.append(span)

            if isinstance(span, MergedSpanAlignment):
                for atomic_span in span.spans:
                    flattened_spans.append(atomic_span)
            else:
                flattened_spans.append(span)

        self.spans = flattened_spans
        self.nested_spans = nested_spans

        source_span_start = min(s.source_span[0] for s in spans)
        source_span_end = max(s.source_span[1] for s in spans)
        target_span_start = min(s.target_span[0] for s in spans)
        target_span_end = max(s.target_span[1] for s in spans)

        self.source_span = (source_span_start, source_span_end)
        self.target_span = (target_span_start, target_span_end)

    def maximal_adjacent_descendant_span_source_distance(self) -> float:
        if len(self.spans) <= 1:
            return 0.

        from models.utils import max_by

        max_dist = float('-inf')
        for span in self.spans:
            other_spans = [
                s for s in self.spans
                if s != span
            ]

            adj_other_span = max_by(other_spans, key=lambda other: -span.source_distance(other))
            dist = span.source_distance(adj_other_span)
            if max_dist < dist:
                max_dist = dist

        return max_dist

    @staticmethod
    def get_minimal_child_span_distance(span_1, span_2):
        spans_1 = span_1.spans if isinstance(span_1, MergedSpanAlignment) else [span_1]
        spans_2 = span_2.spans if isinstance(span_2, MergedSpanAlignment) else [span_2]

        return min(
            s1.distance(s2)
            for s1 in spans_1
            for s2 in spans_2
        )


class SpanAlignmentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SpanAlignment):
            return obj.__dict__

        return json.JSONEncoder.default(self, obj)


def construct_alignment(
    source_tokens: List[str],
    target_tokens: List[str],
    token_alignment: List[Tuple[int, int]],
    merge: bool = False
):
    """
    Args:
        token_alignment: For each source token, its list of aligned tokens in the target side
    """
    span_alignments: List[SpanAlignment] = []

    token_alignment_set = set(token_alignment)
    visited = {align: False for align in token_alignment_set}

    def _get_consecutive_span_alignment(src_idx, tgt_idx):  # noqa
        cur_src_start, cur_src_end = src_idx, src_idx + 1
        cur_tgt_start, cur_tgt_end = tgt_idx, tgt_idx + 1

        visited[(src_idx, tgt_idx)] = True

        def _is_visited(x, y):
            return visited[(x, y)]

        if (src_idx - 1, tgt_idx) in token_alignment_set and not _is_visited(src_idx - 1, tgt_idx):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx - 1, tgt_idx)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        if (src_idx + 1, tgt_idx) in token_alignment_set and not _is_visited(src_idx + 1, tgt_idx):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx + 1, tgt_idx)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        if (src_idx, tgt_idx - 1) in token_alignment_set and not _is_visited(src_idx, tgt_idx - 1):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx, tgt_idx - 1)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        if (src_idx, tgt_idx + 1) in token_alignment_set and not _is_visited(src_idx, tgt_idx + 1):
            (new_src_start, new_src_end), (new_tgt_start, new_tgt_end) = _get_consecutive_span_alignment(src_idx, tgt_idx + 1)

            cur_src_start = min(cur_src_start, new_src_start)
            cur_src_end = max(cur_src_end, new_src_end)
            cur_tgt_start = min(cur_tgt_start, new_tgt_start)
            cur_tgt_end = max(cur_tgt_end, new_tgt_end)

        return (cur_src_start, cur_src_end), (cur_tgt_start, cur_tgt_end)

    for (src_idx, tgt_idx) in token_alignment_set:
        if not visited[(src_idx, tgt_idx)]:
            source_span, target_span = _get_consecutive_span_alignment(src_idx, tgt_idx)

            span_alignments.append(SpanAlignment(source_span, target_span))

    span_alignments = sorted(span_alignments, key=lambda a: (a.source_span[0], a.target_span[0]))

    if merge:
        # merge adjacent bounding boxes
        def _merge_neighboring_spans(span_1: SpanAlignment, span_2: SpanAlignment):
            merged_span = MergedSpanAlignment([span_1, span_2])

            return merged_span

        while True:
            # sort by source index
            span_alignments = sorted(span_alignments, key=lambda a: (a.source_span[0], a.target_span[0]))
            can_merge = False

            for i in range(len(span_alignments) - 1):
                span = span_alignments[i]
                next_span = span_alignments[i + 1]

                dist = span.distance(next_span)
                if dist <= 1 or dist <= math.sqrt(2):
                    can_merge = True

                    if (
                        isinstance(span, MergedSpanAlignment) or
                        isinstance(next_span, MergedSpanAlignment)
                    ):
                        pairwise_dist = MergedSpanAlignment.get_minimal_child_span_distance(span, next_span)
                        if pairwise_dist > math.sqrt(2):
                            can_merge = False

                    if can_merge:
                        merged_span = _merge_neighboring_spans(span, next_span)

                        del span_alignments[i: i + 2]
                        span_alignments.append(merged_span)

                        break

            if not can_merge:
                break

    return span_alignments


def find_covered_span_alignment_heuristic(
    root: Sexp,
    span_alignments: List[SpanAlignment],
    source_tokens: List[str],
    target_tokens: List[str],
    existing_match_results: List[Any]
) -> MergedSpanAlignment:
    """
    Given an S-expression tree `root` and a list of `SpanAlignment`, `span_alignments`,
    return the set of utterance spans aligned to `root`
    """
    flattened_sexp_tokens = sexp_to_seq(root)

    # remove stopword-style leading and ending tokens to increase coverage
    i = 0
    while i < len(flattened_sexp_tokens) and (
        flattened_sexp_tokens[i][0] in {'#', '\'', '"'}
        or flattened_sexp_tokens[i] in {':start', "'?=", 'DateAtTimeWithDefaults', 'Constraint[DateTime]'}
    ):
        i += 1
    start = i
    i = len(flattened_sexp_tokens) - 1
    while i > 0 and flattened_sexp_tokens[i][0] in {'#', '\'', '"'}:
        i -= 1
    end = i + 1
    if end - start > 0:
        flattened_sexp_tokens = flattened_sexp_tokens[start: end]

    flattened_sexp_tokens = canonicalize_plan_tokens(flattened_sexp_tokens)[0]

    try:
        sexp_tokens_start_idx, exp_tokens_end_idx = find_sequence(target_tokens, flattened_sexp_tokens)
    except IndexError:
        print(f'Warning: unable to find {flattened_sexp_tokens} in {target_tokens}')

        return None

    # find overlapping SpanAlignments
    overlapped_alignments = [
        alignment
        for alignment in span_alignments
        if (
            sexp_tokens_start_idx <= alignment.target_span[0] <= exp_tokens_end_idx - 1 or
            sexp_tokens_start_idx <= alignment.target_span[1] - 1 <= exp_tokens_end_idx - 1 or
            alignment.target_span[0] <= sexp_tokens_start_idx <= exp_tokens_end_idx - 1 <= alignment.target_span[1] - 1
        ) and (
            # we only allow single alignment on target tokens whose first letter is alphabetical
            # on it is ":date" since ":date" usually aligns to "at/on"
            alignment.target_span[1] - alignment.target_span[0] > 1 or
            (
                target_tokens[alignment.target_span[0]: alignment.target_span[1]][0][0].isalpha() or
                target_tokens[alignment.target_span[0]: alignment.target_span[1]][0] == ':date'
            )
        ) and (
            not any(
                alignment.source_overlap(m['matched_alignment'])
                for m in existing_match_results
                if m['matched_alignment'] is not None
            )
        )
    ]

    if not overlapped_alignments:
        return None

    overlapped_alignments_bounding_box_start_idx = min(a.target_span[0] for a in overlapped_alignments)
    overlapped_alignments_bounding_box_end_idx = max(a.target_span[1] for a in overlapped_alignments)

    # if (
    #     sexp_tokens_start_idx <= overlapped_alignments_bounding_box_start_idx and
    #     overlapped_alignments_bounding_box_end_idx <= exp_tokens_end_idx
    # ):
    merged_alignment = MergedSpanAlignment(overlapped_alignments)

    if any(
        merged_alignment.source_overlap(m['matched_alignment'])
        for m in existing_match_results
        if m['matched_alignment'] is not None
    ):
        return None

    # if the minimal child span distance is larger than a pre-defined thredshold, we discard this candidate mapping
    min_child_span_distance = merged_alignment.maximal_adjacent_descendant_span_source_distance()
    if min_child_span_distance >= 3:
        return None

    return merged_alignment


def find_matching_tree_node_heuristic(
    root: Sexp,
    span_alignments: List[SpanAlignment],
    source_tokens: List[str],
    target_tokens: List[str],
    parent_arg_name=None,
    include_parent_arg_name_in_match=False,
    matchable_node_names: Optional[Set[str]] = None,
    matchable_parent_node_names: Optional[Set[str]] = None,
    descendant_nodes_can_match: Optional[bool] = None,
    propagate_descendant_match_flag: bool = True,
    allowed_ancestor_nodes: Optional[List[Sexp]] = None,
    existing_match_results: List[Any] = None,
    exclude_arguments: List[str] = None,
    allow_floating_child_derivation: bool = False
) -> Any:
    """
    This procedure uses candidate span-level alignments (without consideration program syntax information)
    to find logically self-contained child nodes on the tree-structured s-expression that can be mapped
    to utterance spans.
    """

    existing_match_results = existing_match_results or []

    node_op_name = None
    if (
        root and
        isinstance(root[0], str)
    ):
        node_op_name = root[0]

    # initialize the flag
    if descendant_nodes_can_match is None:
        if not allowed_ancestor_nodes and not matchable_parent_node_names and not matchable_node_names:
            descendant_nodes_can_match = True
        else:
            descendant_nodes_can_match = False

    descendant_nodes_can_match |= (allowed_ancestor_nodes is not None and root in allowed_ancestor_nodes)

    current_node_can_match = descendant_nodes_can_match
    if matchable_node_names:
        current_node_can_match &= node_op_name in matchable_node_names
    if exclude_arguments:
        current_node_can_match &= all(parent_arg_name != arg for arg in exclude_arguments)

    # check if root could be covered by an alignment
    if current_node_can_match:
        match_result = None

        if parent_arg_name and include_parent_arg_name_in_match:
            match_result = find_covered_span_alignment_heuristic(
                [parent_arg_name, root],
                span_alignments,
                source_tokens,
                target_tokens,
                existing_match_results
            )

        if not match_result:
            match_result = find_covered_span_alignment_heuristic(
                root, span_alignments,
                source_tokens, target_tokens,
                existing_match_results
            )

        if match_result or allow_floating_child_derivation:
            return [{
                'matched_node': root,
                'parent_arg_name': parent_arg_name,
                'matched_alignment': match_result
            }]

    if not propagate_descendant_match_flag:
        descendant_nodes_can_match = False

    if allowed_ancestor_nodes:
        descendant_nodes_can_match = descendant_nodes_can_match or root in allowed_ancestor_nodes
    elif matchable_parent_node_names:
        descendant_nodes_can_match = node_op_name in matchable_parent_node_names
    else:
        descendant_nodes_can_match = True

    # FIXME: # allow matching of nodes rooted at a parent node rooted at `:nonEmptyBase`
    # if parent_arg_name == ':nonEmptyBase':
    #     descendant_nodes_can_match = True

    # get grouped args and values
    grouped_children = []
    if root and isinstance(root[0], str):
        i = 1
    else:
        i = 0

    while i < len(root):
        child = root[i]
        if isinstance(child, str):
            if child.startswith(':'):
                arg_name = child
                arg_value = root[i + 1]
                grouped_children.append({'child': arg_value, 'arg_name': arg_name, 'node_type': 'grouped_argument'})

                i += 2
            elif child == '#':
                arg_name = '#'
                arg_value = root[i + 1]
                grouped_children.append({'child': arg_value, 'arg_name': arg_name, 'node_type': 'literal_value'})

                i += 2
            else:
                # example: let [0]
                # otherwise, skip the string node TODO: check this!
                i += 1
        else:
            # its a value node, do not break it apart
            grouped_children.append({'child': child, 'node_type': 'other'})
            i += 1

    match_results = []
    for child_idx, child in enumerate(grouped_children):
        child_match_results = []

        if child['node_type'] in {'grouped_argument', 'literal_value'}:
            arg_name = child['arg_name']
            arg_val = child['child']

            # remove this spurious case
            # :event StructConstraint[Event]
            child_descendant_nodes_can_match = descendant_nodes_can_match
            if arg_name == ':event':  # and len(arg_val) == 1 and arg_val[0] == 'StructConstraint[Event]'
                child_descendant_nodes_can_match = False

            child_match_results = find_matching_tree_node_heuristic(
                arg_val, span_alignments,
                source_tokens, target_tokens,
                parent_arg_name=arg_name, include_parent_arg_name_in_match=include_parent_arg_name_in_match,
                matchable_node_names=matchable_node_names,
                matchable_parent_node_names=matchable_parent_node_names, allowed_ancestor_nodes=allowed_ancestor_nodes,
                descendant_nodes_can_match=child_descendant_nodes_can_match,
                existing_match_results=match_results + existing_match_results,
                exclude_arguments=exclude_arguments,
                allow_floating_child_derivation=allow_floating_child_derivation
            )  # noqa
        elif child['node_type'] == 'other':
            child_match_results = find_matching_tree_node_heuristic(
                child['child'], span_alignments,
                source_tokens, target_tokens,
                include_parent_arg_name_in_match=include_parent_arg_name_in_match,
                matchable_node_names=matchable_node_names,
                matchable_parent_node_names=matchable_parent_node_names, allowed_ancestor_nodes=allowed_ancestor_nodes,
                descendant_nodes_can_match=descendant_nodes_can_match,
                existing_match_results=match_results + existing_match_results,
                exclude_arguments=exclude_arguments,
                allow_floating_child_derivation=allow_floating_child_derivation
            )

        match_results.extend(child_match_results)

    return match_results


def get_source_target_alignment(
    source_tokens: List[str],
    target_tokens: List[str],
    example: Dict,
    giza_alignment: Dict,
    include_leading_stopword: bool = True,
    output_token_level_alignment: bool = True
) -> Dict:
    # FIXME: was `parse_lispress`
    target_sexp: Sexp = seq_to_sexp(target_tokens)

    alignment = []
    for src_idx, src_tok_align in enumerate(giza_alignment['s2t_alignment']):
        if src_tok_align:
            for tgt_idx in src_tok_align:
                alignment.append((src_idx, tgt_idx))

    assert source_tokens == giza_alignment['source_tokens']

    span_alignments = construct_alignment(
        source_tokens, giza_alignment['target_tokens'],
        alignment,
        merge=False
    )

    match_results = find_matching_tree_node_heuristic(
        target_sexp, span_alignments,
        source_tokens=giza_alignment['source_tokens'],
        target_tokens=giza_alignment['target_tokens'],
        matchable_parent_node_names={
            'Constraint[Event]',
            'EventDuringRange',
            'EventOnDateWithTimeRange',
            'EventOnDateTime',
            'EventOnDateAfterTime',
            'EventOnDate'
            # 'FindEventWrapperWithDefaults'
            # 'String',
            # 'FindManager',
            # 'FindTeamOf',
            # 'FindReports',
            # 'StructConstraint[DateTime]',
            # 'DateAtTimeWithDefaults',
            # 'NumberAM', 'NumberPM'
            # 'HourMinuteAm', 'HourMinutePm', 'HourMilitary', 'HourMinuteMilitary',
            # #'NextDOW',
            # 'LocationKeyphrase',
        },
        # propagate_descendant_match_flag=False,
        include_parent_arg_name_in_match=True
    )

    if len(match_results) == 0:
        match_results = find_matching_tree_node_heuristic(
            target_sexp, span_alignments,
            source_tokens=giza_alignment['source_tokens'],
            target_tokens=giza_alignment['target_tokens'],
            matchable_node_names={
                'FindManager',
                'FindTeamOf',
                'FindReports'
            },
            # propagate_descendant_match_flag=False,
            include_parent_arg_name_in_match=False,
        )

    target_to_source_alignments = []
    for match_idx, result in enumerate(match_results):
        matched_sexp_node: Sexp = result['matched_node']
        source_span_idx = result['matched_alignment'].source_span

        if include_leading_stopword:
            if matched_sexp_node and matched_sexp_node[0] in {
                'FindManager',
                'FindReports',
                'FindTeamOf'
            }:
                # `my manager/boss/team/`
                if source_span_idx[0] - 1 >= 0 and source_tokens[source_span_idx[0] - 1] == 'my':
                    source_span_idx = (source_span_idx[0] - 1, source_span_idx[1])
                    result['matched_alignment'].source_span = source_span_idx

        target_span_idx = result['matched_alignment'].target_span

        matched_source_tokens = source_tokens[source_span_idx[0]: source_span_idx[1]]
        matched_target_sexp_tokens = sexp_to_seq(result['matched_node'])

        if '[0]' in matched_target_sexp_tokens:
            continue

        if result['parent_arg_name']:
            name_arg_val_sexp = [result['parent_arg_name'], matched_sexp_node]

            tokenized_sexp = sexp_to_seq(name_arg_val_sexp)

            try:
                matched_sexp_tokenized_start, matched_sexp_tokenized_end = find_sequence(target_tokens, tokenized_sexp)
            except IndexError:
                # remove last parenthesis
                matched_sexp_tokenized_start, matched_sexp_tokenized_end = find_sequence(target_tokens, tokenized_sexp[1:-1])
        else:
            tokenized_sexp = sexp_to_seq(matched_sexp_node)
            matched_sexp_tokenized_start, matched_sexp_tokenized_end = find_sequence(target_tokens, tokenized_sexp)

        matched_target_tokens = target_tokens[matched_sexp_tokenized_start: matched_sexp_tokenized_end]
        align_entry = {
            'target_tokens_idx': (matched_sexp_tokenized_start, matched_sexp_tokenized_end),
            'target_tokens': matched_target_tokens,
            'source_tokens': matched_source_tokens,
            'source_tokens_idx': tuple(result['matched_alignment'].source_span),
            '__giza_alignment': result['matched_alignment']
        }

        if output_token_level_alignment:
            src_span = result['matched_alignment'].source_span
            tgt_span = result['matched_alignment'].target_span
            token_alignments_in_boundary = [
                (src_idx, tgt_idx)
                for src_idx, tgt_idx in alignment
                if src_span[0] <= src_idx < src_span[1] and tgt_span[0] <= tgt_idx < tgt_span[1]
            ]

            token_alignment_entries = []
            for src_idx, tgt_idx in token_alignments_in_boundary:
                src_token = giza_alignment['source_tokens'][src_idx]
                tgt_token = giza_alignment['target_tokens'][tgt_idx]
                original_tgt_idx = example['canonical_token_index_map'][tgt_idx]

                token_alignment_entries.append({
                    'source_token': src_token,
                    'target_token': tgt_token,
                    'source_token_idx': src_idx,
                    'target_token_idx': original_tgt_idx
                })
                # try:
                #     tgt_idx_in_span = matched_target_tokens.index(tgt_token) + matched_sexp_tokenized_start
                #     token_alignment_entries.append({'source_token_idx': src_idx, 'target_token_idx': tgt_idx_in_span})
                # except ValueError:
                #     continue

            align_entry['token_level_alignments'] = token_alignment_entries

        target_to_source_alignments.append(align_entry)

    output_dict = {
        'alignments': target_to_source_alignments
    }

    if output_token_level_alignment:
        target_to_source_token_alignments = []
        for (src_idx, tgt_idx) in alignment:
            original_tgt_idx = example['canonical_token_index_map'][tgt_idx]
            src_token = giza_alignment['source_tokens'][src_idx]
            tgt_token = giza_alignment['target_tokens'][tgt_idx]
            target_to_source_token_alignments.append({
                'source_token': src_token,
                'target_token': tgt_token,
                'source_token_idx': src_idx,
                'target_token_idx': original_tgt_idx
            })

        output_dict['token_level_alignments'] = target_to_source_token_alignments

    return output_dict


def process_seq2seq_span_alignments(
    dataset_file: Path,
    giza_alignment_file: Path,
    output_file: Path,
    include_leading_stopword: bool = False
):
    giza_alignments: List[Dict] = json.load(giza_alignment_file.open())
    dataset_lines = dataset_file.open().readlines()

    examples = []

    for data_line, giza_alignment in tqdm(
        zip(dataset_lines, giza_alignments), total=len(dataset_lines),
        desc='Generating Alignments...'
    ):
        original_example = json.loads(data_line)
        source_tokens = original_example['utterance'].split(' ')
        target_tokens = original_example['plan'].split(' ')

        example_dict = get_source_target_alignment(
            source_tokens, target_tokens,
            original_example,
            giza_alignment,
            include_leading_stopword=include_leading_stopword,
            output_token_level_alignment=True
        )
        example_dict['source'] = original_example['utterance']
        example_dict['target'] = original_example['plan']
        example_dict['tags'] = original_example['tags']
        # print(align_dict)
        examples.append(example_dict)

    with output_file.open('w') as f:
        for entry in examples:
            f.write(json.dumps(entry, cls=SpanAlignmentEncoder) + '\n')


def get_seq2seq_attention_regularization_data(data_root: Path, args: SimpleNamespace, tag=None):
    for sub_folder in [
        'source_domain_with_target_num8',
        'source_domain_with_target_num16',
        'source_domain_with_target_num32',
        'source_domain_with_target_num64',
        'source_domain_with_target_num128'
    ]:
        data_folder = data_root / sub_folder

        get_giza_alignments(
            data_folder / f's2t_train.VA3.final',
            data_folder / f't2s_train.VA3.final',
            output=data_folder / 'train.giza.alignment'
        )

        output_file = (data_folder / 'train.jsonl').with_suffix(f'.alignment.{tag}.jsonl' if tag else f'.alignment.jsonl')
        process_seq2seq_span_alignments(
            data_folder / 'train.jsonl',
            data_folder / 'train.giza.alignment',
            output_file,
            include_leading_stopword=args.include_leading_stopword
        )

        # split validation data from the joint alignment
        num_train_examples = len((data_folder / 'train.jsonl').open().readlines())
        print(f"Number training instances: {num_train_examples}")
        num_valid_examples = len((data_folder / 'valid.jsonl').open().readlines())
        num_test_examples = len((data_folder / 'test.jsonl').open().readlines())

        valid_va_lines = (data_folder / f's2t_train_eval.VA3.final').open().readlines()[
                         num_train_examples * 3: num_train_examples * 3 + num_valid_examples * 3]
        with (data_folder / f's2t_valid.VA3.final').open('w') as f:
            f.writelines(valid_va_lines)

        valid_va_lines = (data_folder / f't2s_train_eval.VA3.final').open().readlines()[
                         num_train_examples * 3: num_train_examples * 3 + num_valid_examples * 3]
        with (data_folder / f't2s_valid.VA3.final').open('w') as f:
            f.writelines(valid_va_lines)

        get_giza_alignments(
            data_folder / f's2t_valid.VA3.final',
            data_folder / f't2s_valid.VA3.final',
            output=data_folder / 'valid.giza.alignment'
        )

        output_file = (data_folder / 'valid.jsonl').with_suffix(f'.alignment.{tag}.jsonl' if tag else f'.alignment.jsonl')
        process_seq2seq_span_alignments(
            data_folder / 'valid.jsonl',
            data_folder / 'valid.giza.alignment',
            output_file,
            include_leading_stopword=args.include_leading_stopword
        )

        test_va_lines = (data_folder / f's2t_train_eval.VA3.final').open().readlines()[
                         num_train_examples * 3 + num_valid_examples * 3:]
        with (data_folder / f's2t_test.VA3.final').open('w') as f:
            f.writelines(test_va_lines)

        test_va_lines = (data_folder / f't2s_train_eval.VA3.final').open().readlines()[
                         num_train_examples * 3 + num_valid_examples * 3:]
        with (data_folder / f't2s_test.VA3.final').open('w') as f:
            f.writelines(test_va_lines)

        with output_file.with_suffix('.top100.jsonl').open('w') as f:
            f.write(
                ''.join(output_file.open().readlines()[:100])
            )

        get_giza_alignments(
            data_folder / f's2t_test.VA3.final',
            data_folder / f't2s_test.VA3.final',
            output=data_folder / 'test.giza.alignment'
        )

        output_file = (data_folder / 'test.jsonl').with_suffix(f'.alignment.{tag}.jsonl' if tag else f'.alignment.jsonl')
        process_seq2seq_span_alignments(
            data_folder / 'test.jsonl',
            data_folder / 'test.giza.alignment',
            output_file,
            include_leading_stopword=args.include_leading_stopword
        )

        with output_file.with_suffix('.top100.jsonl').open('w') as f:
            f.write(
                ''.join(output_file.open().readlines()[:100])
            )


def rewrite_sexp(root_node: Sexp, target_node: Sexp, new_node: Sexp) -> Tuple[Sexp, bool]:
    if root_node == target_node:
        return new_node, True

    if isinstance(root_node, str):
        return root_node, False

    assert isinstance(root_node, list)

    duplicated_nodes = []
    match_result = False
    for child in root_node:
        new_child, child_match_result = rewrite_sexp(child, target_node, new_node)
        duplicated_nodes.append(new_child)
        match_result = match_result or child_match_result

    return duplicated_nodes, match_result


def extract_sketch_from_alignments(
    source_tokens: List[str],
    target_tokens: List[str],
    example: Dict,
    giza_alignment: Dict,
    allow_floating_child_derivation: bool = False,
    include_leading_stopword: bool = False
) -> Dict:
    target_sexp: Sexp = seq_to_sexp(target_tokens)

    alignment = []
    for src_idx, src_tok_align in enumerate(giza_alignment['s2t_alignment']):
        if src_tok_align:
            for tgt_idx in src_tok_align:
                alignment.append((src_idx, tgt_idx))

    assert source_tokens == giza_alignment['source_tokens']

    span_alignments = construct_alignment(
        source_tokens, giza_alignment['target_tokens'],
        alignment,
        merge=False
    )

    match_results = find_matching_tree_node_heuristic(
        target_sexp, span_alignments,
        source_tokens=giza_alignment['source_tokens'],
        target_tokens=giza_alignment['target_tokens'],
        matchable_parent_node_names={
            'Constraint[Event]',
            'EventDuringRange',
            'EventOnDateWithTimeRange',
            'EventOnDateTime',
            'EventOnDateAfterTime',
            'EventOnDate'
            # 'FindEventWrapperWithDefaults'
            # 'String',
            # 'FindManager',
            # 'FindTeamOf',
            # 'FindReports',
            # 'StructConstraint[DateTime]',
            # 'DateAtTimeWithDefaults',
            # 'NumberAM', 'NumberPM'
            # 'HourMinuteAm', 'HourMinutePm', 'HourMilitary', 'HourMinuteMilitary',
            # #'NextDOW',
            # 'LocationKeyphrase',
        },
        allow_floating_child_derivation=allow_floating_child_derivation,
        # propagate_descendant_match_flag=False,
        include_parent_arg_name_in_match=True,
        exclude_arguments=[
            # ':subject',
            # ':nonEmptyBase'
        ]
    )

    if len(match_results) == 0:
        match_results = find_matching_tree_node_heuristic(
            target_sexp, span_alignments,
            source_tokens=giza_alignment['source_tokens'],
            target_tokens=giza_alignment['target_tokens'],
            matchable_node_names={
                'FindManager',
                'FindTeamOf',
                'FindReports'
            },
            # propagate_descendant_match_flag=False,
            include_parent_arg_name_in_match=False,
            allow_floating_child_derivation=allow_floating_child_derivation
        )

    extracted_sub_trees: Dict[str, Any] = {}
    for match_idx, result in enumerate(match_results):
        matched_target_tokens = sexp_to_seq(result['matched_node'])

        matched_sexp_node: Sexp = result['matched_node']
        tokenized_sexp: List[str] = sexp_to_seq(matched_sexp_node)

        matched_alignment = result['matched_alignment']
        if matched_alignment:
            source_span_idx = result['matched_alignment'].source_span
            target_span_idx = result['matched_alignment'].target_span

            if include_leading_stopword:
                if matched_sexp_node and matched_sexp_node[0] in {
                    'FindManager',
                    'FindReports',
                    'FindTeamOf'
                }:
                    # `my manager/boss/team/`
                    if source_span_idx[0] - 1 >= 0 and source_tokens[source_span_idx[0] - 1] == 'my':
                        source_span_idx = (source_span_idx[0] - 1, source_span_idx[1])
                        matched_alignment.source_span = source_span_idx

            matched_source_tokens = source_tokens[source_span_idx[0]: source_span_idx[1]]
            matched_sexp_tokens = giza_alignment['target_tokens'][target_span_idx[0]: target_span_idx[1]]
        else:
            matched_source_tokens = None
            source_span_idx = target_span_idx = None

        if '[0]' in tokenized_sexp:
            continue

        # print(matched_source_tokens)
        # print(matched_target_tokens)
        # print('--------------')

        slot_name = f'__SLOT{match_idx}__'
        new_sexp_with_slot, match_result = rewrite_sexp(target_sexp, matched_sexp_node, slot_name)

        if match_result is False:
            # if 'DateTimeAndConstraintBetweenEvents' not in target_tokens:
            print(f'WARNING: sub sexp not found for {matched_sexp_node}')
            continue

        target_sexp = new_sexp_with_slot

        sub_tree_start_idx, sub_tree_end_idx = find_sequence(target_tokens, tokenized_sexp)
        extracted_sub_trees[slot_name] = {
            'sub_sexp': ' '.join(tokenized_sexp),
            'parent_arg_name': result['parent_arg_name'],
            'source_tokens_idx': source_span_idx,
            'source_tokens': matched_source_tokens,
            'stripped_target_tokens_idx': target_span_idx,
            'stripped_target_tokens': matched_target_tokens,
            'span_alignment': matched_alignment,
            'sub_tree_start_idx_in_original_target': (sub_tree_start_idx, sub_tree_end_idx)
        }

    root_sexp_sketch_tokens = sexp_to_seq(target_sexp)

    # extract attention regularization data
    alignments = []
    sub_tree_entry: Dict
    for slot_name, sub_tree_entry in extracted_sub_trees.items():
        if sub_tree_entry['span_alignment'] is not None:
            parent_arg_name = sub_tree_entry['parent_arg_name']
            if parent_arg_name:
                query_fragment = [parent_arg_name, slot_name]
            else:
                query_fragment = [slot_name]

            fragment_start_idx, fragment_end_idx = find_sequence(root_sexp_sketch_tokens, query_fragment)
            alignments.append({
                'target_tokens_idx': (fragment_start_idx, fragment_end_idx),
                'target_tokens': root_sexp_sketch_tokens[fragment_start_idx: fragment_end_idx],
                'source_tokens': sub_tree_entry['source_tokens'],
                'source_tokens_idx': tuple(sub_tree_entry['source_tokens_idx'])
            })

    # extract token level alignments
    target_to_source_token_alignments = []
    for (src_idx, tgt_idx) in alignment:
        original_tgt_idx = example['canonical_token_index_map'][tgt_idx]
        src_token = giza_alignment['source_tokens'][src_idx]
        tgt_token = giza_alignment['target_tokens'][tgt_idx]
        assert target_tokens[original_tgt_idx] == tgt_token
        target_to_source_token_alignments.append({
            'source_token': src_token,
            'target_token': tgt_token,
            'source_token_idx': src_idx,
            'target_token_idx': original_tgt_idx
        })

    return {
        'decomposition': {
            'sketch_sexp': ' '.join(root_sexp_sketch_tokens),
            'alignments': alignments,
            'named_sub_sexp': extracted_sub_trees
        },
        'token_level_alignments': target_to_source_token_alignments
    }


def process_structured_span_alignments(
    dataset_file: Path,
    giza_alignment_file: Path,
    output_file: Path,
    allow_floating_child_derivation: bool = False,
    include_leading_stopword: bool = False
):
    giza_alignments: List[Dict] = json.load(giza_alignment_file.open())
    dataset_lines = dataset_file.open().readlines()

    assert len(giza_alignments) == len(dataset_lines)

    examples = []

    for data_line, giza_alignment in tqdm(
        zip(dataset_lines, giza_alignments),
        total=len(dataset_lines), desc='Generating Sketches...'
    ):
        original_example = json.loads(data_line)
        source_tokens = original_example['utterance'].split(' ')
        target_tokens = original_example['plan'].split(' ')

        example_dict = extract_sketch_from_alignments(
            source_tokens, target_tokens,
            original_example,
            giza_alignment,
            allow_floating_child_derivation=allow_floating_child_derivation,
            include_leading_stopword=include_leading_stopword
        )
        example_dict['dialogue_id'] = original_example['dialogue_id']
        example_dict['turn_index'] = original_example['turn_index']
        example_dict['source'] = original_example['utterance']
        example_dict['target'] = original_example['plan']
        example_dict['tags'] = original_example['tags']
        # print(align_dict)
        examples.append(example_dict)

    with output_file.open('w') as f:
        for entry in examples:
            f.write(json.dumps(entry, cls=SpanAlignmentEncoder) + '\n')


def get_structured_parser_attention_regularization_data(data_root: Path, args: SimpleNamespace, tag=''):
    for sub_folder in [
        'source_domain_with_target_num8',
        'source_domain_with_target_num16',
        'source_domain_with_target_num32',
        'source_domain_with_target_num64',
        'source_domain_with_target_num128',
        # 'source_domain_with_target_num256'
    ]:
        data_folder = data_root / sub_folder

        get_giza_alignments(
            data_folder / f's2t_train.VA3.final',
            data_folder / f't2s_train.VA3.final',
            output=data_folder / 'train.giza.alignment'
        )

        output_file = (data_folder / 'train.jsonl').with_suffix(f'.sketch.{tag}.jsonl')
        process_structured_span_alignments(
            data_folder / 'train.jsonl',
            data_folder / 'train.giza.alignment',
            output_file,
            allow_floating_child_derivation=args.allow_floating_child_derivation,
            include_leading_stopword=args.include_leading_stopword
        )

        get_giza_alignments(
            data_folder / f's2t_valid.VA3.final',
            data_folder / f't2s_valid.VA3.final',
            output=data_folder / 'valid.giza.alignment'
        )

        output_file = (data_folder / 'valid.jsonl').with_suffix(f'.sketch.{tag}.jsonl')
        process_structured_span_alignments(
            data_folder / 'valid.jsonl',
            data_folder / 'valid.giza.alignment',
            output_file,
            allow_floating_child_derivation=args.allow_floating_child_derivation,
            include_leading_stopword=args.include_leading_stopword
        )

        with output_file.with_suffix('.top100.jsonl').open('w') as f:
            f.write(
                ''.join(output_file.open().readlines()[:100])
            )

        get_giza_alignments(
            data_folder / f's2t_test.VA3.final',
            data_folder / f't2s_test.VA3.final',
            output=data_folder / 'test.giza.alignment'
        )

        output_file = (data_folder / 'test.jsonl').with_suffix(f'.sketch.{tag}.jsonl')
        process_structured_span_alignments(
            data_folder / 'test.jsonl',
            data_folder / 'test.giza.alignment',
            output_file,
            allow_floating_child_derivation=args.allow_floating_child_derivation,
            include_leading_stopword=args.include_leading_stopword
        )

        with output_file.with_suffix('.top100.jsonl').open('w') as f:
            f.write(
                ''.join(output_file.open().readlines()[:100])
            )


if __name__ == '__main__':
    get_seq2seq_attention_regularization_data(
        Path('data/smcalflow_cs/calflow.orgchart.event_create'),
        SimpleNamespace(
            include_leading_stopword=False
        ),
        tag='release'
    )

    get_structured_parser_attention_regularization_data(
        Path('data/smcalflow_cs/calflow.orgchart.event_create'),
        SimpleNamespace(
            allow_floating_child_derivation=True,
            include_leading_stopword=False
        ),
        tag='release.floating.subtree_idx_info'
    )
