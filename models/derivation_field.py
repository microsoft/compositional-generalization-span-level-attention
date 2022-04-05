from typing import List, Dict, Mapping

from allennlp.data import Vocabulary
from allennlp.nn.util import batch_tensor_dicts
from overrides import overrides
from collections import defaultdict

import torch

from allennlp.data.dataloader import TensorDict
from allennlp.data.fields import Field, ListField, TextField
from allennlp.data.fields.field import DataArray


class DerivationField(Field[DataArray], List[Mapping[str, TensorDict]]):
    __slots__ = ['derivation_levels_named_fields_list']

    empty_fields = {}

    def __init__(self, derivation_levels_named_fields_list: List[Dict[str, Field]]):
        super().__init__()

        self.derivation_levels_named_fields_list = derivation_levels_named_fields_list

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for fields_dict in self.derivation_levels_named_fields_list:
            for field in fields_dict.values():
                field.count_vocab_items(counter)

    @overrides
    def index(self, vocab: Vocabulary):
        for fields_dict in self.derivation_levels_named_fields_list:
            for field in fields_dict.values():
                field.index(vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        deriv_level_fields_dict: Dict[str, Field]
        num_levels = len(self.derivation_levels_named_fields_list)
        padding_lengths = {"num_levels": num_levels}

        for deriv_level, deriv_level_fields_dict in enumerate(self.derivation_levels_named_fields_list):
            for field_name, field in deriv_level_fields_dict.items():
                field_padding_lengths = field.get_padding_lengths()
                for child_key, child_pad_len in field_padding_lengths.items():
                    padding_lengths[f"{deriv_level}::{field_name}::{child_key}"] = child_pad_len

        return padding_lengths

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> List[Dict[str, torch.Tensor]]:  # type: ignore
        num_levels = padding_lengths['num_levels']

        tensor_list = []
        for deriv_level in range(num_levels):
            level_field_keys = [
                key.partition('::')[-1]
                for key in padding_lengths
                if key.startswith(f"{deriv_level}::")
            ]

            level_field_child_padding_lengths = defaultdict(dict)
            for key in level_field_keys:
                field_name, _, child_key = key.partition('::')
                level_field_child_padding_lengths[field_name][child_key] = padding_lengths[
                    f"{deriv_level}::{field_name}::{child_key}"]

            level_tensor_dict = {}

            if deriv_level < len(self.derivation_levels_named_fields_list):
                level_named_fields = self.derivation_levels_named_fields_list[deriv_level]
            else:
                level_named_fields = None

            for field_name, field_padding_lengths in level_field_child_padding_lengths.items():
                if level_named_fields and field_name in level_named_fields:
                    field = level_named_fields[field_name]
                    field_tensor = field.as_tensor(field_padding_lengths)
                else:
                    field_tensor = self.empty_fields[field_name].as_tensor(field_padding_lengths)

                level_tensor_dict[field_name] = field_tensor

            tensor_list.append(level_tensor_dict)

        return tensor_list

    def batch_tensors(self, tensor_list: List[List[Dict[str, torch.Tensor]]]) -> List[DataArray]:
        num_levels = len(tensor_list[0])
        batched_tensor_dict_list = []

        for level in range(num_levels):
            level_fields = list(tensor_list[0][level].keys())
            batched_tensor_dict = {
                field_name: self.empty_fields[field_name].batch_tensors(
                    [
                        x[level][field_name]
                        for x in tensor_list
                    ]
                )
                for field_name
                in level_fields
            }
            batched_tensor_dict_list.append(batched_tensor_dict)

        return batched_tensor_dict_list

    def __len__(self) -> int:
        """
        The length of a derivation is defined as the number of levels
        """

        return len(self.derivation_levels_named_fields_list)
