# pip install python-slugify

import json
import time
from pathlib import Path
import shutil
from slugify import slugify


def run(train_file: str, attention_regularization: str, seed: int = 0, split_name='schema_full_split', batch_size=1, lr=0.001):
    job_name = slugify(
        f'atis_attreg{attention_regularization}_parse_comp_{split_name}_{train_file}_bs{batch_size}_lr{lr}_seed{seed}'
    )

    work_dir = Path(f'runs/') / job_name
    if work_dir.exists():
        if (work_dir / 'metrics.json').exists():
            print(f'{work_dir} is not empty and has results. skipping run {job_name}...')
            return None
        else:
            shutil.rmtree(work_dir)

    work_dir.mkdir(exist_ok=True, parents=True)

    train_file_path = Path(f'data/sql data/atis/{split_name}/{train_file}')
    assert train_file_path.exists()

    extra_config = {
        'pytorch_seed': seed,
        'dataset_reader': {
            'attention_regularization': attention_regularization
        },
        'model': {
            'attention_regularization': attention_regularization
        },
        'train_data_path': f'data/sql data/atis/{split_name}/{train_file}',
        'validation_data_path': f'data/sql data/atis/{split_name}/aligned_final_dev.jsonl',
        'test_data_path': f'data/sql data/atis/{split_name}/final_test.jsonl',
        'iterator': {
            'batch_size': batch_size
        },
        'trainer': {
            'cuda_device': 0,
            'optimizer': {
                'lr': lr
            }
        }
    }
    extra_config_string = json.dumps(extra_config)

    script = f"""
    #!/bin/bash

    conda deactivate
    conda activate allennlp09

    echo Job ID: ${{SLURM_JOB_ID}}>>{work_dir}/slurm.output.txt
    
    allennlp \
        train \
        training_config/iid_ati_covrgseq2seq_attn_reg_elmo_config.jsonnet \
        -s {work_dir} \
        -f \
        --include-package text2sql \
        -o '{extra_config_string}'
    """.strip()

    (work_dir / 'run.sh').open('w').write(script)

    print(script)


if __name__ == '__main__':
    for train_file in [
        'aligned_train.parse_comp_heuristics_false_analyze_nested_false.consecutive_utt.jsonl',
    ]:
        for attn_reg in [
            None,
            'mse:inner:0.05',
            'mse:token:0.05',
            'mse:inner:0.1',
            'mse:token:0.1',
        ]:
            for extra_config in [
                {'batch_size': 4, 'lr': 0.0001},
            ]:
                for seed in range(5):
                    run(
                        train_file,
                        seed=seed,
                        attention_regularization=attn_reg,
                        split_name='new_question_split',
                        **extra_config
                    )
