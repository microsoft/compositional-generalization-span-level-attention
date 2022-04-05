import argparse
import json

import torch
from pathlib import Path
from types import SimpleNamespace
import pytorch_lightning as pl

try:
    from .attention_regularized_seq2seq import AttentionRegularizedTranslationModule
except:
    from attention_regularized_seq2seq import AttentionRegularizedTranslationModule


def run_eval(args: SimpleNamespace):
    model = AttentionRegularizedTranslationModule.load_from_checkpoint(
        str(args.model_path),
        data_dir=args.data_path,
        eval_batch_size=20,
        n_train=0,
        n_test=args.num_examples
    )

    if args.beam_size:
        model.eval_beams = model.hparams.eval_beams = args.beam_size

    if torch.cuda.is_available():
        model = model.to('cuda')
        gpus = 1
    else:
        gpus = 0

    trainer = pl.Trainer(gpus=gpus)

    result_dict = trainer.test(
        model,
        model.get_dataloader('test', model.hparams.eval_batch_size),
    )

    output = {}
    for k, v in result_dict.items():
        if k in ["log", "progress_bar"]:
            continue

        if torch.is_tensor(v):
            v = v.item()

        output[k] = v

    if args.save_path:
        with args.save_path.open('w') as f:
            f.write(json.dumps(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument('--data_path', required=True, type=Path)
    parser.add_argument('--save_path', required=False, type=Path, default=None)
    parser.add_argument('--num_examples', type=int, default=-1)
    parser.add_argument('--beam_size', type=int, default=None)

    args = parser.parse_args()
    run_eval(args)


if __name__ == '__main__':
    main()
