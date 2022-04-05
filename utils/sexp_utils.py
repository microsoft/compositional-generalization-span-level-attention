from typing import List, TypeVar, Any, Tuple

from dataflow.core.sexp import Sexp

T = TypeVar('T')


def flatten(nested_list: List[List[T]]) -> List[T]:
    return [
        element
        for sub_list in nested_list
        for element in sub_list
    ]


def sexp_to_tokenized_string(sexp: Sexp) -> List[str]:
    """
    Shamelessly borrowed from dataflow
    Generates tokenized string representation from S-expression
    """
    if isinstance(sexp, list):
        return ['('] + flatten([sexp_to_tokenized_string(f) for f in sexp]) + [')']
    else:
        return [sexp]


def sexp_to_string(sexp: Sexp) -> str:
    return ' '.join(sexp_to_tokenized_string(sexp))


def find_sequence(sequence: List[Any], query: List[Any]) -> Tuple[int, int]:
    sll = len(query)
    for ind in (i for i, e in enumerate(sequence) if e == query[0]):
        if sequence[ind:ind + sll] == query:
            return ind, ind + sll

    raise IndexError


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
