import math
import os, sys
from pathlib import Path

on_azure = os.environ.get('ON_AZURE', False)
if on_azure:
    print('Initializing wandb logger...')
    os.system('wandb login && wandb on')

    # need the parent dir module
    sys.path.insert(2, str(Path(__file__).resolve().parents[1]))

import wandb
import argparse
import glob
import json
import os
import time
import traceback
from collections import defaultdict
from typing import Union, List, Iterable, Optional, Dict, Tuple

import numpy as np

import torch
import torch.nn as nn

from allennlp.data import Instance, Token
import allennlp.nn.util as allennlp_util
from allennlp.data.dataloader import TensorDict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.tokenizers import pretrained_transformer_tokenizer as allennlp_transformer_tokenizer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from transformers import AutoTokenizer, BatchEncoding, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import Seq2SeqLMOutput

try:
    from .lightning_base import generic_train, BaseTransformer, add_generic_args
    from .callbacks import get_early_stopping_callback, Seq2SeqLoggingCallback, get_checkpoint_callback
    from .finetune import SummarizationModule, TranslationModule
    from .utils import lmap, flatten_list, pickle_save, label_smoothed_nll_loss, freeze_embeds, assert_all_frozen, get_git_info, freeze_params
except ImportError:
    from lightning_base import generic_train, BaseTransformer, add_generic_args
    from callbacks import get_early_stopping_callback, Seq2SeqLoggingCallback, get_checkpoint_callback
    from finetune import SummarizationModule, TranslationModule
    from utils import lmap, flatten_list, pickle_save, label_smoothed_nll_loss, freeze_embeds, assert_all_frozen, get_git_info, freeze_params


class AttentionRegularizedSeq2SeqDatasetReader(DatasetReader):
    def __init__(
        self,
        model_name: str,
        attention_regularization: str = None,
        num_examples: int = -1,
        example_tags: Optional[list] = None,
        target_token_canonicalization_dict: Optional[Path] = None
    ):
        super().__init__()

        self.allennlp_tokenizer = allennlp_transformer_tokenizer.PretrainedTransformerTokenizer(model_name)
        self.attention_regularization = attention_regularization
        self.attention_regularization_config: Optional[Dict] = None
        if attention_regularization:
            self.attention_regularization_config = \
                AttentionRegularizedTranslationModule._parse_attention_regularization(attention_regularization)  # noqa

        self.num_examples = num_examples

        if example_tags:
            print(f'Will only load examples with tags: {example_tags}')
        self.example_tags = example_tags

        self.target_token_canonicalization_dict = None
        if target_token_canonicalization_dict:
            token_dict = {}
            for line in target_token_canonicalization_dict.open():
                data = line.strip().split('\t')
                assert len(data) == 2
                assert data[0] not in token_dict
                assert all(
                    data[1] != other_val
                    for other_val
                    in token_dict.values()
                )
                token_dict[data[0]] = data[1]

            self.target_token_canonicalization_dict = token_dict

    def read_instances(
        self,
        file_path: Union[Path, str],
    ) -> List[Instance]:
        dataset = super().read(file_path)
        # noqa
        return dataset.instances

    def _read(self, file_path: str) -> Iterable[Instance]:
        idx = 0
        for line in open(file_path):
            data = json.loads(line)
            tags = data.get('tags', [])

            is_valid_example = True
            if self.example_tags:
                if not any(tag in self.example_tags for tag in tags):
                    is_valid_example = False

            if is_valid_example:
                yield self.to_instance(
                    data['source'], data['target'],
                    data['alignments'], data.get('token_level_alignments'),
                    tags=tags,
                    example_idx=data['example_idx']
                )
            else:
                continue

            idx += 1
            if self.num_examples is not None and 0 < self.num_examples == idx:
                break

    def to_instance(
        self,
        source: str,
        target: str,
        alignments: Optional[Dict] = None,
        token_level_alignments: Optional[Dict] = None,
        example_idx: Optional[int] = None,
        tags: List[str] = None,
    ) -> Instance:
        source = source.lower()
        target = target.lower()
        source_tokens, source_subtoken_offsets = self.allennlp_tokenizer.intra_word_tokenize(source.split(' '))

        target_tokens_by_whitespace: List[str] = target.split(' ')
        target_token_offsets = [(idx, idx) for idx, token in enumerate(target_tokens_by_whitespace)]
        if self.target_token_canonicalization_dict:
            canonical_target_token_offsets = []
            canonical_target_tokens = []
            for token_idx, token in enumerate(target_tokens_by_whitespace):
                if token in self.target_token_canonicalization_dict:
                    canonical_tokens = self.target_token_canonicalization_dict[token].split(' ')
                    offset = (len(canonical_target_tokens), len(canonical_target_tokens) + len(canonical_tokens) - 1)
                else:
                    canonical_tokens = [token]
                    offset = (len(canonical_target_tokens), len(canonical_target_tokens))

                canonical_target_tokens.extend(canonical_tokens)
                canonical_target_token_offsets.append(offset)
        else:
            canonical_target_tokens = target_tokens_by_whitespace
            canonical_target_token_offsets = target_token_offsets

        canonical_target = ' '.join(canonical_target_tokens)

        # assert '_' not in canonical_target
        # assert '.' not in canonical_target

        target_tokens, canonical_target_subtoken_offsets = self.allennlp_tokenizer.intra_word_tokenize(
            canonical_target_tokens)

        target_subtoken_offsets = []
        for token_idx, token in enumerate(target_tokens_by_whitespace):
            canonical_token_offset = canonical_target_token_offsets[token_idx]
            subtoken_start_offset = canonical_target_subtoken_offsets[canonical_token_offset[0]]
            subtoken_end_offset = canonical_target_subtoken_offsets[canonical_token_offset[1]]
            original_tgt_token_subtoken_offset = (subtoken_start_offset[0], subtoken_end_offset[1])
            target_subtoken_offsets.append(original_tgt_token_subtoken_offset)

        # sanity check
        t5_tokenizer: T5Tokenizer = self.allennlp_tokenizer.tokenizer
        # source_tokens_ = self.allennlp_tokenizer.tokenizer(source)
        # target_tokens_ = self.allennlp_tokenizer.tokenizer.tokenize(target)

        # assert allennlp_source_tokens_ == source_tokens_
        # assert allennlp_target_tokens_ == target_tokens_

        # add preceeding <pad> for target tokens
        # target_tokens.insert(0, Token(t5_tokenizer.pad_token, text_id=t5_tokenizer.pad_token_id))
        # target_subtoken_offsets = self.allennlp_tokenizer._increment_offsets(target_subtoken_offsets, 1) # noqa

        allennlp_source_token_ids_ = [tok.text_id for tok in source_tokens]
        allennlp_target_token_ids_ = [tok.text_id for tok in target_tokens]

        seq2seq_batch = t5_tokenizer.prepare_seq2seq_batch([source], [canonical_target], truncation=False)
        seq2seq_src_token_ids = seq2seq_batch['input_ids'][0]
        seq2seq_tgt_token_ids = seq2seq_batch['labels'][0]

        assert seq2seq_src_token_ids == allennlp_source_token_ids_
        assert seq2seq_tgt_token_ids == allennlp_target_token_ids_

        # reconstructed_tgt = t5_tokenizer.decode(seq2seq_tgt_token_ids, clean_up_tokenization_spaces=False)
        # print(reconstructed_tgt)

        fields = {
            'source': source,
            'target': canonical_target,
            'source_tokens': source_tokens,
            'target_tokens': target_tokens,
            'tags': tags,
            'example_idx': example_idx
        }

        if self.target_token_canonicalization_dict:
            fields['original_target'] = target

        if self.attention_regularization and alignments:
            alignment_padding_val = -1
            align_matrix = np.zeros((len(target_tokens), len(source_tokens)))
            align_mat_view = align_matrix
            alignment_spans = []
            use_token_level_alignment = ':token:' in self.attention_regularization
            use_span_level_alignment = not use_token_level_alignment

            if use_span_level_alignment:
                for entry in alignments:
                    tgt_token_start, tgt_token_end = entry['target_tokens_idx']
                    src_token_start, src_token_end = entry['source_tokens_idx']

                    src_span_subtokens, (src_subtoken_start, src_subtoken_end) = self.get_subtokens_slice(
                        source_tokens,
                        (src_token_start, src_token_end), source_subtoken_offsets
                    )

                    tgt_span_subtokens, (tgt_subtoken_start, tgt_subtoken_end) = self.get_subtokens_slice(
                        target_tokens,
                        (tgt_token_start, tgt_token_end), target_subtoken_offsets
                    )

                    # print(f'{" ".join(target_tokens_by_whitespace[tgt_token_start:tgt_token_end])} ---> {target_tokens[tgt_subtoken_start: tgt_subtoken_end]}')
                    # print("************")

                    alignment_spans.append({
                        'tgt_span': (tgt_subtoken_start, tgt_subtoken_end),
                        'src_span': (src_subtoken_start, src_subtoken_end)
                    })

                    if ':inner:' in self.attention_regularization or ':all:' in self.attention_regularization:
                        # align_mat_view is wrapped by <s> and </s>
                        if 'complementary' in self.attention_regularization:
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

                            if 'boundary_decay' in self.attention_regularization_config:
                                assert self.attention_regularization_config['src_normalize']
                                eta = self.attention_regularization_config['boundary_decay']
                                wnd_size = 3

                                for idx in range(max(0, src_token_start - wnd_size), src_token_start):
                                    dist = src_token_start - idx
                                    align_mat_view[
                                        tgt_subtoken_start: tgt_subtoken_end,
                                        idx
                                    ] = math.exp(-eta * dist)

                                for idx in range(src_subtoken_end, min(len(source_tokens), src_subtoken_end + wnd_size)):
                                    dist = idx - src_subtoken_end + 1
                                    align_mat_view[
                                        tgt_subtoken_start: tgt_subtoken_end,
                                        idx
                                    ] = math.exp(-eta * dist)

                if ':outer:' in self.attention_regularization or ':all:' in self.attention_regularization:
                    source_token_alignment_mask_for_sketch_token = np.ones(len(source_tokens))

                    if 'complementary' in self.attention_regularization:
                        source_token_alignment_mask_for_sketch_token.fill(-1)

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
            elif use_token_level_alignment:
                # assert use_token_level_regularization
                for entry in token_level_alignments:
                    source_span, (src_subtoken_start, src_subtoken_end) = self.get_subtokens_slice(
                        source_tokens,
                        (entry['source_token_idx'], entry['source_token_idx'] + 1), source_subtoken_offsets
                    )

                    target_span, (tgt_subtoken_start, tgt_subtoken_end) = self.get_subtokens_slice(
                        target_tokens,  # account for <s> and </s>
                        (entry['target_token_idx'], entry['target_token_idx'] + 1), target_subtoken_offsets
                    )

                    align_mat_view[
                        tgt_subtoken_start: tgt_subtoken_end,
                        src_subtoken_start: src_subtoken_end
                    ] = 1.0

            for i in range(align_mat_view.shape[0]):
                if sum(align_mat_view[i]) == 0:
                    align_mat_view[i] = alignment_padding_val

            fields['target_to_source_alignment'] = align_matrix

        return Instance(fields)

    def get_subtokens_slice(
        self,
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


class AttentionRegularizedSeq2SeqDataset(Dataset):
    def __init__(
        self,
        instances: List[Instance],
        tokenizer: AutoTokenizer,
        max_source_length: Optional[int] = None,
        max_target_length: Optional[int] = None
    ):
        self.instances = instances
        self.tokenizer = tokenizer

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.vocab = None  # to be compatible with Allennlp

    def __iter__(self):
        return iter(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def __len__(self):
        return len(self.instances)

    def collate(self, examples: List[Instance]) -> TensorDict:
        batch_encoding: BatchEncoding = self.tokenizer.prepare_seq2seq_batch(
            [
                example["source"]
                for example
                in examples
            ],
            tgt_texts=[
                example["target"]
                for example
                in examples
            ],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
        )

        # print(batch_encoding['labels'].shape)
        batch_tensors = batch_encoding.data

        if 'target_to_source_alignment' in examples[0]:
            max_target_length = max(e['target_to_source_alignment'].shape[0] for e in examples)
            max_source_length = max(e['target_to_source_alignment'].shape[1] for e in examples)
            alignment_tensor = np.full((len(examples), max_target_length, max_source_length), fill_value=-1, dtype=np.float32)

            for example_id, example in enumerate(examples):
                example_alignment = example['target_to_source_alignment']
                alignment_tensor[example_id, :example_alignment.shape[0], :example_alignment.shape[1]] = example_alignment

            batch_tensors['target_to_source_alignment'] = torch.from_numpy(alignment_tensor)

        metadata = {
            'tags': [e['tags'] for e in examples],
            'source': [e['source'] for e in examples],
            'target': [e['target'] for e in examples]
        }
        batch = batch_encoding.data
        batch['metadata'] = metadata

        return batch


class AttentionRegularizedTranslationModule(BaseTransformer):
    mode = "translation"
    loss_names = ["loss", "attention_regularization_loss"]
    metric_names = ["acc"]
    default_val_metric = "acc"

    def __init__(self, hparams: argparse.Namespace, **kwargs):
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)

        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        # save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        # pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }

        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }

        # assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        # assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"

        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        self.hparams.git_sha = get_git_info()["repo_sha"]
        self.hparams.pwd = os.getcwd()
        self.num_workers = hparams.num_workers
        self.decoder_start_token_id = None  # default to config
        self.already_saved_batch = False
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams

        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length

        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        self.model.decoder.config.output_attentions = True
        attention_regularization_config_string = getattr(hparams, 'attention_regularization', None)
        self.attention_regularization = self._parse_attention_regularization(
            getattr(hparams, 'attention_regularization', None))
        self.attention_regularization_config_string = attention_regularization_config_string

    @staticmethod
    def _parse_attention_regularization(config_string: str):
        config_dict = {}
        if config_string:
            config_dict = {
                'src_normalize': False,
                'complementary': False
            }

            data = config_string.split(':')
            # mse:all:1-4|-1:4.0
            for entry_idx, entry in enumerate(data):
                if entry in {'mse'}:
                    config_dict['type'] = entry
                elif entry in {'all', 'inner', 'outer', 'token'}:
                    config_dict['span_type'] = entry
                elif entry == 'src_normalize':
                    config_dict['src_normalize'] = True
                elif entry == 'complementary':
                    config_dict['complementary'] = True
                elif entry.startswith('boundary_decay'):
                    decay_val = float(entry.partition('boundary_decay_')[-1])
                    config_dict['boundary_decay'] = decay_val
                elif entry.startswith('example_tags['):
                    assert entry.endswith(']')
                    tags = entry.partition('[')[-1].rpartition(']')[0]
                    tags = tags.strip().split(',')
                    config_dict['example_tags'] = set(tags)
                elif entry_idx < len(data) - 1:
                    layer_ids = [int(idx) for idx in entry.split(',')]
                    config_dict['layer_ids'] = layer_ids
                elif entry_idx == len(data) - 1:
                    assert float(entry) >= 0.
                    config_dict['weight'] = float(entry)

            if config_dict['src_normalize'] and config_dict['complementary']:
                raise ValueError(f'Config error: {config_dict}')

        return config_dict

    def get_dataset(self, type_path) -> AttentionRegularizedSeq2SeqDataset:
        max_target_length = self.target_lens[type_path]

        example_tags = self.hparams.get('example_tags')
        if example_tags:
            example_tags = example_tags.split(',')

        dataset_reader = AttentionRegularizedSeq2SeqDatasetReader(
            self.hparams.model_name_or_path,
            self.attention_regularization_config_string if type_path == 'train' else None,
            num_examples=-1 if (self.hparams.shuffle_val and type_path == 'val') else self.n_obs[type_path],
            example_tags=example_tags,
            target_token_canonicalization_dict=self.hparams.target_token_canonicalization_dict
        )

        if type_path == 'val':
            type_path = 'dev'

        instances = dataset_reader.read_instances(Path(self.hparams.data_dir) / type_path / f'{type_path}.jsonl')
        print(f'Loaded {len(instances)} {type_path} examples in total, tags: {example_tags}')
        dataset = AttentionRegularizedSeq2SeqDataset(
            instances,
            self.tokenizer,
            max_source_length=self.hparams.max_source_length, max_target_length=max_target_length
        )

        # print('Avg. num. of target tokens: ', np.average([len(inst['target_tokens']) for inst in dataset.instances]))
        # print('Max. num. of target tokens: ', max([len(inst['target_tokens']) for inst in dataset.instances]))
        #
        # if type_path == 'train':
        #     with open(Path(self.hparams.output_dir) / 'train.tgt', 'w') as f:
        #         for inst in dataset.instances:
        #             f.write(' '.join([token.text for token in inst['target_tokens']]) + '\n')

        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        num_examples = self.n_obs['val']
        if type_path == 'val' and self.hparams.shuffle_val:
            rng = np.random.RandomState(seed=1234)
            idx_list = list(range(len(dataset)))
            rng.shuffle(idx_list)
            if num_examples > 0:
                idx_list = idx_list[:num_examples]
            dataset.instances = [dataset.instances[idx] for idx in idx_list]

        # generator = torch.Generator(self.device)
        # generator.manual_seed(self.hparams.seed * 577)

        if type_path == 'train':
            batch_sampler = BucketBatchSampler(
                dataset,
                batch_size=batch_size,
                sorting_keys=['target_tokens']
            )

            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate,
                num_workers=self.num_workers
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate,
                shuffle=shuffle,
                num_workers=self.num_workers,
            )

        return dataloader

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)
        # if not self.already_saved_batch:  # This would be slightly better if it only happened on rank zero
        #     batch["decoder_input_ids"] = decoder_input_ids
        #     self.save_readable_batch(batch)

        outputs = self(
            src_ids,
            attention_mask=src_mask, decoder_input_ids=decoder_input_ids,
            use_cache=False,
            target_to_source_alignment=batch.get('target_to_source_alignment'),
            metadata=batch['metadata']
        )

        lm_logits = outputs[0]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )

        loss_tuple = (loss, )

        if self.training and self.attention_regularization:
            attn_reg_loss = outputs[-1]
            loss = loss + self.attention_regularization['weight'] * attn_reg_loss
            loss_tuple = (loss, attn_reg_loss)

        return loss_tuple

    def forward(self, input_ids, **kwargs):
        metadata = kwargs.pop('metadata', None)
        target_to_source_alignment = kwargs.pop('target_to_source_alignment', None)

        model_output: Tuple = self.model(input_ids, **kwargs)

        # for entry in model_output:
        #     print(type(entry))
        #     if torch.is_tensor(entry):
        #         print(entry.shape)

        if self.attention_regularization and self.training:
            example_tags = self.attention_regularization.get('example_tags')
            do_regularization = True
            example_mask = None
            if example_tags:
                example_mask = [
                    (
                        True
                        if any(tag in example_tags for tag in e_tags)
                        else False
                    )
                    for e_tags
                    in metadata['tags']
                ]
                if not any(example_mask):
                    do_regularization = False
                else:
                    # (batch_size,)
                    examples_mask_tensor = torch.tensor(example_mask, dtype=torch.bool).to(self.device)

            if do_regularization:
                decoder_attentions = model_output[1][1::2]
                attention_regularization_losses = []
                for layer_id in self.attention_regularization['layer_ids']:
                    # (batch_size, num_heads, sequence_length, sequence_length)
                    decoder_attention_weights = decoder_attentions[layer_id]
                    model_output_target_sequence_length = decoder_attention_weights.size(2)

                    # shape: (batch_size, target_sequence_length, source_sequence_length)
                    target_to_source_alignment_mask = (target_to_source_alignment != -1)
                    # shape: (batch_size, target_sequence_length, source_sequence_length)
                    target_to_source_alignment = target_to_source_alignment * target_to_source_alignment_mask

                    # print('T5 att:', decoder_attention_weights.shape)
                    # print('Alignment: ', target_to_source_alignment.shape)

                    # if target_to_source_alignment.size(1) < model_output_target_sequence_length:
                    #     decoder_attention_weights = decoder_attention_weights[:, :, :model_output_target_sequence_length, :]

                    num_heads = decoder_attention_weights.size(1)

                    # shape: (batch_size, target_sequence_length - 1, source_sequence_length)
                    target_to_source_alignment_trimmed = target_to_source_alignment[:, 1:]
                    # shape: (batch_size, 1, target_sequence_length - 1, source_sequence_length)
                    target_to_source_alignment_mask_trimmed = target_to_source_alignment_mask[:, None, 1:]
                    # shape: (batch_size, num_heads, target_sequence_length - 1, source_sequence_length)
                    decoder_attention_weights_trimmed = decoder_attention_weights[:, :, :-1]

                    tgt_sequence_length = target_to_source_alignment_trimmed.size(1)
                    decoder_tgt_sequence_length = decoder_attention_weights_trimmed.size(2)

                    if tgt_sequence_length > decoder_tgt_sequence_length:
                        target_to_source_alignment_trimmed = target_to_source_alignment_trimmed[:, :decoder_tgt_sequence_length]
                        target_to_source_alignment_mask_trimmed = target_to_source_alignment_mask_trimmed[:, :, :decoder_tgt_sequence_length]

                    # shape: (batch_size, trimmed_target_sequence_length, source_sequence_length)
                    target_attention_distribution = target_to_source_alignment_trimmed
                    target_attention_distribution_mask = target_to_source_alignment_mask_trimmed.squeeze(1)
                    # shape: (batch_size, trimmed_target_sequence_length)
                    target_attention_distribution_target_timestep_mask = target_attention_distribution_mask.any(dim=-1)

                    if self.attention_regularization['src_normalize']:
                        target_attention_distribution = target_attention_distribution / (
                            target_attention_distribution.sum(dim=-1) +
                            1e-10 * ~target_attention_distribution_target_timestep_mask
                        ).unsqueeze(-1)

                    # try:
                    # shape: (batch_size, num_heads, target_sequence_length - 1, source_sequence_length)
                    attention_regularization_loss_layer_i = nn.MSELoss(reduction='none')(
                        decoder_attention_weights_trimmed,
                        target_attention_distribution.unsqueeze(1).expand(-1, num_heads, -1, -1)
                    ) * target_to_source_alignment_mask_trimmed
                    # except Exception as e:
                    #     print(decoder_attention_weights_trimmed.shape)
                    #     print(target_to_source_alignment_trimmed.shape)
                    #     print(metadata)
                    #
                    #     raise e

                    # shape: (batch_size, )
                    num_attn_regularized_tgt_tokens = target_attention_distribution_target_timestep_mask.sum(dim=-1)

                    # shape: (batch_size, num_heads)
                    attention_regularization_loss_layer_i = (
                        attention_regularization_loss_layer_i
                        .sum(dim=-1)
                        .sum(dim=-1)
                    ) / (
                        num_attn_regularized_tgt_tokens.unsqueeze(1) +
                        allennlp_util.tiny_value_of_dtype(torch.float)
                    )

                    # shape: (batch_size,)
                    attention_regularization_loss_layer_i = attention_regularization_loss_layer_i.mean(dim=-1)
                    if example_mask:
                        # shape: (1, )
                        attention_regularization_loss_layer_i = allennlp_util.masked_mean(
                            attention_regularization_loss_layer_i,
                            examples_mask_tensor,  # noqa
                            dim=-1
                        )
                    else:
                        # shape: (1, )
                        # over all heads and batch examples
                        attention_regularization_loss_layer_i = attention_regularization_loss_layer_i.mean()

                    attention_regularization_losses.append(attention_regularization_loss_layer_i)

                attention_regularization_loss = torch.stack(attention_regularization_losses).mean()
            else:
                attention_regularization_loss = 0.

            model_output = model_output + (attention_regularization_loss,)

        return model_output

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {
            name: loss
            for name, loss
            in zip(self.loss_names, loss_tensors)
        }

        # tokens per batch
        logs["tokens_per_batch"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["batch_size"] = batch["input_ids"].shape[0]
        logs["num_source_pad_tokens"] = batch["input_ids"].eq(self.pad).sum()
        logs["source_pad_tokens_frac"] = batch["input_ids"].eq(self.pad).float().mean()

        # TODO(SS): make a wandb summary metric for this
        return {
            "loss": loss_tensors[0],
            "log": logs
        }

    def _convert_ids_to_tokens(self, batched_token_ids: List[List[int]]):
        return self.tokenizer.batch_decode(batched_token_ids, clean_up_tokenization_spaces=False)

    def _generative_step(self, batch: dict, batch_idx: int = 0) -> dict:
        t0 = time.time()

        with torch.no_grad():
            generated_ids = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=True,
                decoder_start_token_id=self.decoder_start_token_id,
                num_beams=self.eval_beams,
                # max_length=self.eval_max_length,
                max_length=self.hparams.val_max_target_length
            )

            gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
            preds: List[str] = self._convert_ids_to_tokens(generated_ids)
            untruncated_target_token_ids = self.tokenizer.prepare_seq2seq_batch(
                batch['metadata']['source'],
                batch['metadata']['target'],
                max_target_length=10000
            )['labels']
            target: List[str] = self._convert_ids_to_tokens(untruncated_target_token_ids)

            if batch_idx < 10:
                input_ids_array = batch['input_ids'].cpu().numpy()
                for idx in range(len(input_ids_array)):
                    source = self.tokenizer.decode(list(input_ids_array[idx]), skip_special_tokens=False)
                    print(f'Example {idx}: {source}')
                    print(f'Pred: {preds[idx]}')
                    print(f'Tgt: {target[idx]}')

            if self.val_metric == 'loss':
                loss_tensors = self._step(batch)
                base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
            else:
                base_metrics = {}
            # print(loss_tensors)
            #
            # max_len = max(len(x) for x in generated_ids)
            # targets = np.zeros((len(generated_ids), max_len), dtype=np.int64)
            # for example_id, pred_token_ids in enumerate(generated_ids):
            #     targets[example_id, :len(pred_token_ids) - 1] = pred_token_ids[1:]  # skip leading <pad_id>
            #
            # print(targets)
            # eval_batch = dict(batch)
            # eval_batch['labels'] = torch.from_numpy(targets)
            # print(self._step(eval_batch))

            summ_len = np.mean(lmap(len, generated_ids))
            base_metrics.update(
                gen_time=gen_time,
                gen_len=summ_len,
                preds=preds,
                target=target,
                tags=batch['metadata'].get('tags', [])
            )

        return base_metrics

    def validation_step(self, batch, batch_idx) -> Dict:
        if self.val_metric == 'loss':
            loss_tensors = self._step(batch)
            logs = {
                name: loss
                for name, loss
                in zip(self.loss_names, loss_tensors)
            }

            return logs
        elif self.val_metric == 'acc':
            return self._generative_step(batch, batch_idx)
        else:
            raise ValueError(self.val_metric)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1

        run_generation = self.val_metric == 'acc' or prefix == 'test'
        generative_metrics = {}
        losses = {}

        if run_generation:
            predictions = flatten_list([x['preds'] for x in outputs])
            targets = flatten_list([x['target'] for x in outputs])
            metric_val = acc = self.calculate_acc(predictions, targets)
            metric_name = 'acc'
            generative_metrics[metric_name] = metric_val

            if outputs and len(outputs[0].get('tags', [])) > 0:
                # num examples
                example_tags: List[List[str]] = flatten_list([x['tags'] for x in outputs])
                assert len(example_tags) == len(predictions)

                unique_tags = sorted(set(flatten_list(example_tags)))
                for tag in unique_tags:
                    examples_with_tag = [
                        (hyp, tgt)
                        for hyp, tgt, tags
                        in zip(predictions, targets, example_tags)
                        if tag in tags
                    ]
                    tag_i_acc = self.calculate_acc([x[0] for x in examples_with_tag], [x[1] for x in examples_with_tag])
                    generative_metrics[f'tag_{tag}_acc'] = tag_i_acc

        elif self.val_metric == 'loss':
            metric_name = 'loss'

            losses = {
                k: torch.stack([x[k] for x in outputs]).mean()
                for k in
                self.loss_names
                if k in outputs[0]
            }

            metric_val = losses['loss']
            generative_metrics[metric_name] = metric_val
        else:
            raise ValueError(f'Unknown validation metric {self.val_metric}')

        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).float()
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)

        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path

        output_dict = {
            "log": all_metrics,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

        if self.val_metric == 'loss':
            output_dict[f"{prefix}_loss"] = losses['loss']

        if run_generation:
            output_dict['predictions'] = predictions  # noqa
            output_dict['targets'] = targets  # noqa

        return output_dict

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def calculate_acc(self, predictions, targets):
        results = [
            1
            if pred.strip().lower() == tgt.strip().lower()
            else 0
            for pred, tgt
            in zip(predictions, targets)
        ]

        acc = np.average(results)

        return acc

    def calc_generative_metrics(self, preds, target) -> dict:
        return {
            'acc': self.calculate_acc(preds, target)
        }

    def train_dataloader(self) -> DataLoader:
        # dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        # `train_loader` is already initialized in super().setup()
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser: argparse.ArgumentParser, root_dir: str):
        SummarizationModule.add_model_specific_args(parser, root_dir)
        parser.add_argument('--example_tags', required=False, default=None, type=str)
        parser.add_argument("--shuffle_val", required=False, action='store_true')
        parser.add_argument('--attention_regularization', type=str, default=None)
        parser.add_argument('--target_token_canonicalization_dict', type=Path, default=None)

        return parser


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = AttentionRegularizedTranslationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    # if len([file for file in os.listdir(args.output_dir) if 'slurm' not in file]) > 3 and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    dataset_root_folder = Path(args.data_dir).name

    model = AttentionRegularizedTranslationModule(args)

    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = TensorBoardLogger(
            save_dir=model.output_dir,
            name='tensorboard_logs'
        )
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset_root_folder)
        print(f'wandb {wandb.__version__} project name: {project}')
        logger = WandbLogger(project=project)

        while True:
            try:
                logger._experiment = wandb.init(project=project)
                break
            except:
                traceback.print_exc()
                print("Retrying....")
                time.sleep(10)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset_root_folder}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    with open(model.output_dir / 'config.json', 'w') as f:
        json.dump(model.hparams, f, indent=2, default=str)

    lower_is_better = args.val_metric == "loss"
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=logger,
    )

    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == '__main__':
    main()
