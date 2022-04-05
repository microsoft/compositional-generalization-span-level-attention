from typing import List, Any


def find_sequence(sequence: List[Any], query: List[Any]):
    sll = len(query)
    for ind in (i for i, e in enumerate(sequence) if e == query[0]):
        if sequence[ind:ind + sll] == query:
            return ind, ind + sll

    raise IndexError
