import json
from pathlib import Path
from utils.cfq.rewrite_sparql import rewrite, SimplifyFunction
from utils.train_giza_alignment_general import train_giza_bidirectional


def run_alignment(src_file: Path, tgt_file: Path, label: str, preprocess_only: bool = False, rewrite_level: SimplifyFunction = None):
    tgt_file_name_without_suffix = tgt_file.name[:-len(tgt_file.suffix)]
    simplified_tgt_file = tgt_file.with_name(f'{tgt_file_name_without_suffix}_{label}.txt')

    with simplified_tgt_file.open('w') as f, tgt_file.with_suffix(f'.{label}.meta.jsonl').open('w') as f_meta:
        for line in tgt_file.open():
            query = line.strip()

            if rewrite_level is not None:
                rewritten_query, metadata = rewrite(query, rewrite_level, return_metadata=True)
            else:
                rewritten_query = query
                metadata = {}

            f.write(rewritten_query + '\n')
            f_meta.write(json.dumps(metadata) + '\n')

    if not preprocess_only:
        train_giza_bidirectional(src_file, simplified_tgt_file, label=label)


def main():
    dataset_root = Path('data/cfq/').expanduser()
    simplify_func = SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS
    # simplify_func = None
    label = 'simplified'
    for split in ['mcd1', 'mcd2', 'mcd3']:
        run_alignment(
            dataset_root / split / 'train' / 'train_encode.txt',
            dataset_root / split / 'train' / 'train_decode.txt',
            label=label,
            rewrite_level=simplify_func
        )

        run_alignment(
            dataset_root / split / 'dev' / 'dev_encode.txt',
            dataset_root / split / 'dev' / 'dev_decode.txt',
            label=label,
            rewrite_level=simplify_func,
            preprocess_only=True
        )

        run_alignment(
            dataset_root / split / 'test' / 'test_encode.txt',
            dataset_root / split / 'test' / 'test_decode.txt',
            label=label,
            rewrite_level=simplify_func,
            preprocess_only=True
        )


if __name__ == '__main__':
    main()
