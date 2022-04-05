import collections
from functools import partial
from typing import Dict, Iterable, Callable, List
from pathlib import Path
import copy


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def default_param_sweep(config: Dict) -> Iterable[Dict]:
    yield config


def get_param_sweep_for_bert_models(config: Dict, **kwargs):
    config = copy.deepcopy(config)
    for config in get_param_sweep_for_bert_encoder(config, **kwargs):
        for config in get_param_sweep_for_att_reg(config):
            config = copy.deepcopy(config)

            yield config


def get_param_sweep_for_decompositional_soft_span_select(config: Dict, **kwargs):
    config = copy.deepcopy(config)

    for config in get_param_sweep_for_bert_encoder(
        config,
        batch_size=32,
        gradient_accumulation_steps=2,
        **kwargs
    ):
        for span_mask_weight in [None]:
            config.setdefault('model', {}).setdefault('attention', {})['span_mask_weight'] = span_mask_weight
            if span_mask_weight is None:
                config['model']['root_derivation_compute_span_scores'] = False
            else:
                config['model']['root_derivation_compute_span_scores'] = True

            for sketch_level_att_reg in [False]:  # True,
                config.setdefault('model', {})['sketch_level_attention_regularization_only'] = sketch_level_att_reg
                for config in get_param_sweep_for_att_reg(config):
                    yield copy.deepcopy(config)


def get_param_sweep_for_bert_encoder(config: Dict, **kwargs):
    config = copy.deepcopy(config)

    train_file = Path(kwargs['local_train_file'])
    train_file = train_file.expanduser()
    assert train_file.exists()
    num_train_examples = len(train_file.open().readlines())

    batch_size = kwargs.get('batch_size', 64)
    gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
    num_epoch = kwargs.get('num_epoch', None)
    if num_epoch is None:
        num_epoch = config.get('trainer', {}).get('num_epochs', 30)

    total_train_steps = num_train_examples * num_epoch / batch_size / gradient_accumulation_steps
    lrs = kwargs.get('lr', [3e-5])
    for lr in lrs:
        config.setdefault('trainer', {})['num_epochs'] = num_epoch
        (
            config
            .setdefault('data_loader', {})
            .setdefault('batch_sampler', {})
            ['batch_size']
        ) = batch_size
        config['trainer']['num_gradient_accumulation_steps'] = gradient_accumulation_steps
        config['trainer'].setdefault('optimizer', {})['lr'] = lr

        warm_up_steps = config['trainer'].setdefault('learning_rate_scheduler', {}).get(
            'warmup_steps',
            int(total_train_steps * 0.1)
        )
        config['trainer'].setdefault('learning_rate_scheduler', {}).update({
            'warmup_steps': warm_up_steps,
            # 'total_steps': int(total_train_steps)
        })

        yield config


def get_param_sweep_for_att_reg(config: Dict, **kwargs) -> Iterable[Dict]:
    config = copy.deepcopy(config)
    att_reg_methods = [None]
    for att_reg in ['mse']:
        for att_reg_pattern in ['token:src_normalize', 'all:src_normalize']:  # 'complementary'
            for att_reg_weight in [0.1, 0.5, 1.0, 2.0, 4.0]:
                att_reg_method = f'{att_reg}:{att_reg_pattern}:{att_reg_weight}'
                att_reg_methods.append(att_reg_method)

    for att_reg_method in att_reg_methods:
        config.setdefault('model', {})['attention_regularization'] = att_reg_method
        config.setdefault('dataset_reader', {})['attention_regularization'] = att_reg_method

        yield copy.deepcopy(config)


run_groups = {
    'calflow.orgchart.event_create': {
        'models': {
            'seq2seq.bert': {
                'train_file': 'train.alignment.jsonl',
                'valid_file': 'valid.alignment.jsonl',
                'test_file': 'test.alignment.jsonl',
                'param_sweep_function': partial(
                    get_param_sweep_for_bert_models,
                    batch_size=32,
                    gradient_accumulation_steps=2,
                )
            },
            'structured.bert': {
                'config_file': 'config.calflow.structured.bert.jsonnet',
                'train_file': 'train.sketch.floating.subtree_idx_info.jsonl',
                'valid_file': 'valid.sketch.floating.subtree_idx_info.jsonl',
                'test_file': 'test.sketch.floating.subtree_idx_info.jsonl',
                'extra_config': {
                    'model': {
                        "child_derivation_compute_span_scores": False,
                        "root_derivation_compute_span_scores": False
                    }
                },
                'param_sweep_function': get_param_sweep_for_decompositional_soft_span_select
            },
        }
    },
}
