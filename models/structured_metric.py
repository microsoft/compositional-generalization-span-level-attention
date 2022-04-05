from typing import Iterable, Any, Dict, List, Set, Optional, Tuple
from overrides import overrides
from collections import defaultdict

from dataflow.core.sexp import Sexp

from allennlp.training.metrics import Metric


def extract_sketch_from_logical_form(
    root: Sexp,
    parent_arg_name=None,
    matchable_node_names: Optional[Set[str]] = None,
    matchable_parent_node_names: Optional[Set[str]] = None,
    descendant_nodes_can_match: Optional[bool] = None,
    propagate_descendant_match_flag: bool = True,
    allowed_ancestor_nodes: Optional[List[Sexp]] = None,
    exclude_arguments: List[str] = None,
    existing_child_derivations: List[Dict] = None
) -> Sexp:
    # base case
    if isinstance(root, str):
        return root

    node_op_name = None

    if (
        root and
        isinstance(root, list) and
        isinstance(root[0], str)
    ):
        node_op_name = root[0]

    # initialize the flag
    if descendant_nodes_can_match is None:
        if (
            not allowed_ancestor_nodes and
            not matchable_parent_node_names and
            not matchable_node_names
        ):
            descendant_nodes_can_match = True
        else:
            descendant_nodes_can_match = False

    descendant_nodes_can_match |= (
        allowed_ancestor_nodes is not None and
        root in allowed_ancestor_nodes
    )

    current_node_can_match = descendant_nodes_can_match
    if matchable_node_names:
        current_node_can_match &= node_op_name in matchable_node_names
    if exclude_arguments:
        current_node_can_match &= all(
            parent_arg_name != arg
            for arg in exclude_arguments
        )

    existing_child_derivations = existing_child_derivations or []

    # check if root could be covered by an alignment
    if current_node_can_match:
        child_derivation_id = len(existing_child_derivations)
        slot_name = f'__SLOT{child_derivation_id}__'

        return slot_name

    if not propagate_descendant_match_flag:
        descendant_nodes_can_match = False

    if allowed_ancestor_nodes:
        descendant_nodes_can_match |= root in allowed_ancestor_nodes
    elif matchable_parent_node_names:
        descendant_nodes_can_match = node_op_name in matchable_parent_node_names
    else:
        descendant_nodes_can_match = True

    # get grouped args and values
    replaced_child_nodes = []
    child_node_ptr = 0
    while child_node_ptr < len(root):
        child_node: Sexp = root[child_node_ptr]
        if isinstance(child_node, str) and child_node.startswith(':'):
            # current child node is an argument name
            arg_name = child_node
            replaced_child_nodes.append(arg_name)

            if child_node_ptr + 1 < len(root):
                arg_val = root[child_node_ptr + 1]

                replaced_arg_val = extract_sketch_from_logical_form(
                    arg_val,
                    parent_arg_name=arg_name,
                    matchable_node_names=matchable_node_names,
                    matchable_parent_node_names=matchable_parent_node_names,
                    descendant_nodes_can_match=descendant_nodes_can_match and arg_name != ':event',
                    propagate_descendant_match_flag=propagate_descendant_match_flag,
                    allowed_ancestor_nodes=allowed_ancestor_nodes,
                    exclude_arguments=exclude_arguments,
                    existing_child_derivations=existing_child_derivations
                )

                replaced_child_nodes.append(replaced_arg_val)

            child_node_ptr += 2
        else:
            replaced_child = extract_sketch_from_logical_form(
                child_node,
                matchable_node_names=matchable_node_names,
                matchable_parent_node_names=matchable_parent_node_names,
                descendant_nodes_can_match=descendant_nodes_can_match,
                propagate_descendant_match_flag=propagate_descendant_match_flag,
                allowed_ancestor_nodes=allowed_ancestor_nodes,
                exclude_arguments=exclude_arguments,
                existing_child_derivations=existing_child_derivations
            )

            replaced_child_nodes.append(replaced_child)
            child_node_ptr += 1

    return replaced_child_nodes


@Metric.register('structured')
class StructuredRepresentationMetric(Metric):
    def __init__(self) -> None:
        self._counters = {}
        self.reset()

    @staticmethod
    def get_sketch_level_em(hyp_program: Sexp, ref_program: Sexp) -> bool:
        if isinstance(hyp_program, str) and hyp_program.startswith('__SLOT') and isinstance(ref_program, list):
            return True

        if isinstance(hyp_program, str) or isinstance(ref_program, str):
            return hyp_program == ref_program

        # if len(hyp_program) == 1 and isinstance(hyp_program[0], str) and hyp_program[0].startswith('__SLOT'):
        #     return True

        # both are s-expression
        if len(hyp_program) != len(ref_program):
            return False

        for node_idx in range(len(ref_program)):
            hyp_node = hyp_program[node_idx]
            ref_node = ref_program[node_idx]

            node_match_result = StructuredRepresentationMetric.get_sketch_level_em(hyp_node, ref_node)
            if node_match_result is False:
                return False

        return True

    def get_sketch_and_child_subtree_em(
        self,
        hyp_program: Sexp,
        hyp_sketch: Optional[Sexp],
        ref_program: Sexp
    ) -> Tuple[bool, bool, Dict]:
        if hyp_sketch is None:
            hyp_sketch = extract_sketch_from_logical_form(
                hyp_program,
                matchable_parent_node_names={
                    'StructConstraint[Event]',
                    'EventDuringRange',
                    'EventOnDateWithTimeRange',
                    'EventOnDateTime',
                    'EventOnDateAfterTime',
                    'EventOnDate'
                },
                exclude_arguments=[':subject', ':nonEmptyBase']
            )

        child_subtree_match_results = []

        def find(_hyp_program_node: Sexp, _hyp_sketch_node: Sexp, _ref_program_node: Sexp):
            if isinstance(_hyp_sketch_node, str) and _hyp_sketch_node.startswith('__SLOT'):
                child_subtree_match = _hyp_program_node == _ref_program_node
                slot_name = _hyp_sketch_node

                child_subtree_match_results.append({
                    'slot_name': slot_name,
                    'hyp_subtree': _hyp_program_node,
                    'ref_subtree': _ref_program_node,
                    'is_correct': child_subtree_match
                })

                return None
            # remaining cases are sketch-level cases
            elif isinstance(_hyp_program_node, str) or isinstance(_ref_program_node, str):
                return _hyp_program_node == _ref_program_node
            else:
                if not (len(_hyp_program_node) == len(_ref_program_node) == len(_hyp_sketch_node)):
                    return False

                for node_idx in range(len(_ref_program_node)):
                    hyp_node = _hyp_program_node[node_idx]
                    hyp_sketch_node = _hyp_sketch_node[node_idx]
                    ref_node = _ref_program_node[node_idx]

                    node_match_result = find(hyp_node, hyp_sketch_node, ref_node)
                    if node_match_result is False:
                        return False

                return True

        sketch_em = bool(find(hyp_program, hyp_sketch, ref_program))
        subtree_em = all(x['is_correct'] for x in child_subtree_match_results)

        return sketch_em, subtree_em, child_subtree_match_results

    @staticmethod
    def get_em_without_matching_node(hyp_program: Sexp, ref_program: Sexp, exclude_nodes: List[str] = None) -> bool:
        exclude_nodes = exclude_nodes or []

        if (
            isinstance(hyp_program, list) and
            isinstance(ref_program, list) and
            len(hyp_program) > 0 and
            len(ref_program) > 0 and
            isinstance(hyp_program[0], str) and
            isinstance(ref_program[0], str) and
            ref_program[0] == hyp_program[0] and
            ref_program[0] in exclude_nodes
        ):
            return True

        if isinstance(hyp_program, str) or isinstance(ref_program, str):
            return hyp_program == ref_program

        # both are s-expression
        if len(hyp_program) != len(ref_program):
            return False

        for node_idx in range(len(ref_program)):
            hyp_node = hyp_program[node_idx]
            ref_node = ref_program[node_idx]

            node_match_result = StructuredRepresentationMetric.get_em_without_matching_node(hyp_node, ref_node, exclude_nodes)
            if node_match_result is False:
                return False

        return True

    @overrides
    def __call__(
        self,
        hyp_programs: List[Sexp],
        hyp_sketches: Optional[List[Sexp]],
        ref_programs: List[Sexp],
        tag_set_lists: List[Iterable[str]] = None
    ):
        tag_set_lists = tag_set_lists or [{'all'}] * len(hyp_programs)
        if hyp_sketches is None:
            hyp_sketches = [None] * len(hyp_programs)

        for hyp_program, hyp_sketch, ref_program, tags in zip(hyp_programs, hyp_sketches, ref_programs, tag_set_lists):
            # self._total_em += hyp_program == ref_program
            # hyp_sketch = hyp_derivation['representation']
            if hyp_sketch is not None:
                sketch_em = float(self.get_sketch_level_em(hyp_sketch, ref_program))
            else:
                sketch_em = 0.

            hyp_abstract_sketch = extract_sketch_from_logical_form(
                hyp_program,
                matchable_parent_node_names={
                    'StructConstraint[Event]',
                    # 'Constraint[Event]',
                    'EventDuringRange',
                    'EventOnDateWithTimeRange',
                    'EventOnDateTime',
                    'EventOnDateAfterTime',
                    'EventOnDate'
                },
                exclude_arguments=[':subject', ':nonEmptyBase']
            )
            abstract_sketch_em = float(self.get_sketch_level_em(hyp_abstract_sketch, ref_program))

            _sketch_em, subtree_em, subtree_match_results = self.get_sketch_and_child_subtree_em(
                hyp_program,
                hyp_sketch if hyp_sketch is not None else hyp_abstract_sketch,
                ref_program
            )

            subtree_em &= _sketch_em

            em_without_string_literal = StructuredRepresentationMetric.get_em_without_matching_node(
                hyp_program, ref_program,
                ['String']
            )

            em_without_string_and_nextdow = StructuredRepresentationMetric.get_em_without_matching_node(
                hyp_program, ref_program,
                ['String', 'NextDOW']
            )

            for tag in tags:
                self._counters.setdefault(tag, defaultdict(int))['sketch_em'] += sketch_em
                self._counters.setdefault(tag, defaultdict(int))['abstract_sketch_em'] += abstract_sketch_em
                self._counters[tag]['child_subtree_em'] += float(subtree_em)

                self._counters[tag]['without_string_literal_em'] += em_without_string_literal
                self._counters[tag]['without_string_and_nextdow_em'] += em_without_string_and_nextdow

                self._counters.setdefault(tag, defaultdict(int))['count'] += 1

            # print('*' * 10)
            # print('Sketch', hyp_sketch)
            # print('Program', ref_program)
            # print('Sketch Match', sketch_em)
            # self._sketch_em += sketch_em
            # self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        for tag, metric in self._counters.items():
            for metric_name in (n for n in metric if n != 'count'):
                metric_val = metric[metric_name] / metric['count'] if metric['count'] > 0 else 0.

                metrics.update({
                    f'{tag}_{metric_name}': metric_val,
                    f'{tag}_count': metric['count']
                })

        if reset:
            self.reset()

        return metrics

    @overrides
    def reset(self):
        self._counters = {}

    def __str__(self):
        return f"SequenceMatchingMetric({self.get_metric()})"

