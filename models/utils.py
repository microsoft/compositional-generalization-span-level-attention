import json
from pathlib import Path
from typing import List, Dict, Any, Callable, TypeVar, Tuple, Union, Optional, Hashable

import torch
from scipy.special import logsumexp

from allennlp.data import TextFieldTensors, Token
from allennlp.data.dataloader import TensorDict
from allennlp.training import BatchCallback
from dataflow.core.lispress import parse_lispress
from dataflow.core.sexp import Sexp


T = TypeVar('T')


@BatchCallback.register('batch_logger')
class BatchLogger(BatchCallback):
    def __call__(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool = False,
    ) -> None:
        if 'loss' in batch_outputs[0]:
            # if batch_number % 1 == 0:
            print(f"batch {batch_number} loss={batch_outputs[0]['loss'].item()}")


def first(sequence: List[T], func: Callable[[T], bool]) -> Optional[T]:
    for item in sequence:
        if func(item):
            return item

    return None


def group_by_and_return_indices(sequence: List[T], key_func: Callable[[T], Hashable]) -> List[List[int]]:
    item_ids_with_same_key = dict()

    for item_id, item in enumerate(sequence):
        item_ids_with_same_key.setdefault(key_func(item), []).append(item_id)

    return [v for v in item_ids_with_same_key.values()]


def log_sum_exp(input: List[float]) -> float:
    return logsumexp(input)


def find_sequence(sequence: List[Any], query: List[Any]):
    query_seq_len = len(query)
    for idx in (
        i
        for i, tok
        in enumerate(sequence)
        if tok == query[0]
    ):
        if sequence[idx:idx + query_seq_len] == query:
            return idx, idx + query_seq_len

    raise IndexError


def replace_with_sequence(sequence: List[Any], query_sequence: List[Any], target_sequence: List[Any]):
    query_seq_start, query_seq_end = find_sequence(sequence, query_sequence)
    new_sequence = sequence[:query_seq_start] + target_sequence + sequence[query_seq_end:]

    return new_sequence


def text_field_tensor_apply(
    text_field_tensor: TextFieldTensors,
    fn: Callable[[torch.Tensor], torch.Tensor]
) -> TextFieldTensors:
    return {
        indexer_name: {
            name: fn(value)
            for name, value
            in named_tensors.items()
        }
        for indexer_name, named_tensors
        in text_field_tensor.items()
    }


def flatten_text_field_tensor(text_field_tensor: TextFieldTensors, start_dim: int, end_dim: int) -> TextFieldTensors:
    new_tensor_dict = {}
    for indexer_name, tensors in text_field_tensor.items():
        new_tensor_dict[indexer_name] = {}
        for tensor_name, tensor in tensors.items():
            new_tensor_dict[indexer_name][tensor_name] = tensor.flatten(start_dim, end_dim)

    return new_tensor_dict


def flatten_tensor_dict(tensor_dict: Dict[str, torch.Tensor], start_dim: int, end_dim: int) -> Dict[str, torch.Tensor]:
    return {
        key: val.flatten(start_dim, end_dim) if torch.is_tensor(val) else val
        for key, val
        in tensor_dict.items()
    }


def flatten(nested_list: List[List[T]]) -> List[T]:
    return [
        element
        for sub_list in nested_list
        for element in sub_list
    ]


def sexp_to_tokenized_str(sexp: Sexp) -> List[str]:
    """
    Shamelessly borrowed from dataflow
    Generates tokenized string representation from S-expression
    """
    if isinstance(sexp, list):
        return ['('] + flatten([sexp_to_tokenized_str(f) for f in sexp]) + [')']
    else:
        return [sexp]


def max_by(elements: List[T], key: Callable[[T], float]) -> T:
    best_element_pos = None
    best_element_score = float('-inf')

    if len(elements) == 0:
        raise ValueError('Size of list is zero')

    for i, element in enumerate(elements):
        score_i = key(element)
        if score_i > best_element_score:
            best_element_pos = i
            best_element_score = score_i

    return elements[best_element_pos]


def load_jsonl_file(file_path: Union[str, Path]) -> List[Dict]:
    file_path = Path(file_path)

    return [
        json.loads(line)
        for line
        in file_path.open()
    ]


def tokenize_sexp_logical_form(
    sexp_string: str,
) -> Tuple[List[Token], Dict]:
    sexp = parse_lispress(sexp_string)

    utterance_span_mentions = []
    subtoken_offsets: List[Tuple[int, int]] = []

    def sexp_to_tokenized_str(sexp: Sexp, start_idx=0) -> Tuple[List[Token], int]:
        if isinstance(sexp, list):
            if (
                len(sexp) > 0 and
                isinstance(sexp[0], str) and
                sexp[0] in {'String', 'LocationKeyphrase', 'PersonName'}
            ):
                assert sexp[1] == sexp[-1] == '"', f'Error when parsing literals in {sexp_string}'

                string_tokens = sexp[2:-1]
                token_offsets = [(i + start_idx + 3, i + start_idx + 3) for i in range(len(string_tokens))]

                sexp_tokens = [
                    Token('('),
                    Token(sexp[0]),
                    Token(sexp[1])
                ] + [
                    Token(token)
                    for token
                    in string_tokens
                ] + [
                    Token(sexp[-1]),
                    Token(')')
                ]

                subtoken_offsets.extend([
                    (start_idx, start_idx),
                    (start_idx + 1, start_idx + 1),
                    (start_idx + 2, start_idx + 2)
                ] + token_offsets + [
                    (start_idx + 3 + len(string_tokens),
                     start_idx + 3 + len(string_tokens)),
                    (start_idx + 3 + len(string_tokens) + 1,
                     start_idx + 3 + len(string_tokens) + 1),
                ])

                utterance_span_mentions.append(
                    (start_idx + 3, start_idx + 3 + len(string_tokens))
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

    tokenized_target_sexp, end_idx = sexp_to_tokenized_str(sexp)

    metadata = {
        'sexp': sexp,
        'utterance_span_mentions': utterance_span_mentions,
        'subtoken_offsets': subtoken_offsets
    }

    return tokenized_target_sexp, metadata