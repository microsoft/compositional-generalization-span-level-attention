"""
Running experiments on SMCalFlow

Usage:
    submit.py MODEL_NAMES --dataset-prefix=<path> [options]

Options:
    -h --help                               show this screen.
    -f                                      Override
    --dry-run                               Dry run
    --local                                 If it is a local run
    --domain=<text>                         Domain [default: calflow]
    --seeds=<text>                          Seeds [default: 0,]
    --label=<text>                          Run label [default: null]
    --dataset-prefix=<path>                  Dataset prefix
    --no-cuda                               Run on cpu
"""

import contextlib
import copy
import datetime
import subprocess
import sys
import json
import time
from pathlib import Path
import collections.abc
from typing import Dict, Iterable

import os
from docopt import docopt
from subprocess import call

from utils.calflow.experiment_configs import *


args = docopt(__doc__)


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def ensure_list(val):
    return val if isinstance(val, list) else [val]


if __name__ == '__main__':
    model_names = args['MODEL_NAMES'].split(',')
    run_label = args['--label']
    overwrite = args['-f']
    dryrun = args['--dry-run']
    domain = args['--domain']
    domain_info = run_groups[domain]
    seeds = [int(seed) for seed in args['--seeds'].strip().split(',') if seed is not None and seed != '']
    # tgt_nums = [16, 32, 64, 128]
    dataset_path_prefix = Path(args['--dataset-prefix'])
    no_cuda = args['--no-cuda']

    assert dataset_path_prefix.exists()

    if run_label.strip() == 'null':
        run_label = 'run'

    run_label += '_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S%f")

    for model_name in model_names:
        model_config: Dict = domain_info['models'][model_name]
        config_file = Path('data') / 'configs' / model_config.get(
            'config_file',
            f'config.calflow.{model_name}.deploy.jsonnet'
        )
        assert config_file.exists()

        extra_config = model_config.get('extra_config', {})

        param_sweep_func = model_config.get('param_sweep_function', default_param_sweep)

        local_train_file = dataset_path_prefix / model_config.get("train_file")

        assert local_train_file.exists()

        sweeped_configs = param_sweep_func(
            extra_config,
            domain_name=domain,
            local_train_file=local_train_file
        )

        for run_group_idx, extra_config_dict in enumerate(sweeped_configs):
            extra_config_dict['train_data_path'] = dataset_path_prefix / model_config['train_file']
            extra_config_dict['validation_data_path'] = dataset_path_prefix / model_config['valid_file']
            extra_config_dict['test_data_path'] = dataset_path_prefix / model_config['test_file']
            extra_config_dict['evaluate_on_test'] = True

            for seed in seeds:
                job_name = f"""{run_label}_{model_name}_run{run_group_idx}_seed{seed}"""
                job_name = job_name.replace('"', '').replace(':', '_').replace('.', '_').replace('-', '_')
                work_dir = Path(f'runs/{job_name}')

                if work_dir.exists():
                    if overwrite:
                        os.system(f'rm -rf {work_dir}')
                    else:
                        raise RuntimeError(f'work dir {work_dir} already exists!')

                if not dryrun:
                    work_dir.mkdir(parents=True)

                print(f"Running training procedure at folder {work_dir}")

                config_dict = {
                    'wandb': False,
                    'pytorch_seed': seed,
                    'trainer': {
                        'patience': 10,
                        'validation_metric': '+events_with_orgchart_em',
                        'cuda_device': -1 if no_cuda else 0
                    }
                }

                update(config_dict, extra_config_dict)

                command = (
                    f"""CUBLAS_WORKSPACE_CONFIG=:16:8 """ +
                    f"""TRANSFORMERS_CACHE=/tmp/ ALLENNLP_CACHE_ROOT=/tmp/ """ +
                    f"""allennlp train {config_file} --file-friendly-logging -s {work_dir} --include-package models """ +
                    f"""-o '{json.dumps(config_dict, default=str)}'"""
                )

                script = f"""#!/bin/bash
        
    {command}
    """.strip()

                if not dryrun:
                    with (work_dir / 'run.sh').open('w') as f:
                        f.write(script)

                    os.chmod(work_dir / 'run.sh', 0o755)

                    subprocess.call(str(work_dir / 'run.sh'))
                else:
                    print(f'Ready to submit the following job script @ {work_dir}')
                    print(script)
