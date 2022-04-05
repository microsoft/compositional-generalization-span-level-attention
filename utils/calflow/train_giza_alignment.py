"""
Train GIZA alignment

Usage:
    train_giza_alignment.py DATA_PATH

"""

import sh
from pathlib import Path
import os
from docopt import docopt

args = docopt(__doc__)


def run(data_root: Path):
    GIZA_ROOT = Path('./giza-pp/GIZA++-v2').expanduser().absolute()
    assert GIZA_ROOT.exists()

    data_root = data_root.resolve()

    for data_folder_name in [
        'source_domain_with_target_num8',
        'source_domain_with_target_num16',
        'source_domain_with_target_num32',
        'source_domain_with_target_num64',
        'source_domain_with_target_num128'
    ]:
        data_folder = data_root / data_folder_name
        os.chdir(data_folder)
        os.system(f"{GIZA_ROOT / 'plain2snt.out'} train.src train.canonical.tgt")

        os.chdir(GIZA_ROOT)
        os.system(
            f"./trainGIZA++.sh "
            f"{data_folder / 'train.src.vcb'} "
            f"{data_folder / 'train.canonical.tgt.vcb'} "
            f"{data_folder / 'train.src_train.canonical.tgt.snt'} "
            f"s2t_train"
        )
        os.system(f'cp s2t_train.VA3.final {data_folder}')

        os.chdir(data_folder)
        os.system(f"{GIZA_ROOT / 'plain2snt.out'} train.canonical.tgt train.src")

        os.chdir(GIZA_ROOT)
        os.system(
            f"./trainGIZA++.sh "
            f"{data_folder / 'train.canonical.tgt.vcb'} "
            f"{data_folder / 'train.src.vcb'} "
            f"{data_folder / 'train.canonical.tgt_train.src.snt'} "
            f"t2s_train"
        )
        os.system(f'cp t2s_train.VA3.final {data_folder}')

        os.chdir(data_folder)
        os.system(f"cat train.src valid.src test.src > train_eval.src")
        os.system(f"cat train.canonical.tgt valid.canonical.tgt test.canonical.tgt > train_eval.canonical.tgt")
        os.system(f"{GIZA_ROOT / 'plain2snt.out'} train_eval.src train_eval.canonical.tgt")

        os.chdir(GIZA_ROOT)
        os.system(
            f"./trainGIZA++.sh "
            f"{data_folder / 'train_eval.src.vcb'} "
            f"{data_folder / 'train_eval.canonical.tgt.vcb'} "
            f"{data_folder / 'train_eval.src_train_eval.canonical.tgt.snt'} "
            f"s2t_train_eval"
        )
        os.system(f'cp s2t_train_eval.VA3.final {data_folder}')

        os.chdir(data_folder)
        os.system(f"{GIZA_ROOT / 'plain2snt.out'} train_eval.canonical.tgt train_eval.src")

        os.chdir(GIZA_ROOT)
        os.system(
            f"./trainGIZA++.sh "
            f"{data_folder / 'train_eval.canonical.tgt.vcb'} "
            f"{data_folder / 'train_eval.src.vcb'} "
            f"{data_folder / 'train_eval.canonical.tgt_train_eval.src.snt'} "
            f"t2s_train_eval"
        )
        os.system(f'cp t2s_train_eval.VA3.final {data_folder}')


if __name__ == '__main__':
    run(
        data_root=Path(args['DATA_PATH'])
    )
