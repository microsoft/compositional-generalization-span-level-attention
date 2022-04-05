from pathlib import Path
import json
from typing import Dict
import numpy as np


def is_correct(example: Dict) -> bool:
    # return example['predicted_tokens'] == example['metadata']['target_tokens']
    return float(example['metadata']['is_prediction_correct'])


if __name__ == '__main__':

    for decode_file_name in ['decode.test.jsonl', 'decode.dev.jsonl']:
        results = {}

        for run_root_template in [
            'runs/atis-attregmse-inner-0-05-parse-comp-aligned-train-parse-comp-heuristics-false-analyze-nested-false-consecutive-utt-jsonl-seed{seed}',
            'runs/atis-attregmse-token-0-05-parse-comp-aligned-train-parse-comp-heuristics-false-analyze-nested-false-consecutive-utt-jsonl-seed{seed}'
        ]:
            for seed in range(0, 5):
                work_dir = Path(run_root_template.format(seed=seed))
                decode_file = work_dir / decode_file_name

                assert decode_file.exists()

                decode_results = [
                    json.loads(line)
                    for line
                    in decode_file.open()
                ]

                eval_list = [
                    is_correct(e)
                    for e
                    in decode_results
                ]

                acc = np.average(eval_list)
                print(decode_file_name)
                print(work_dir)
                print(f'Acc={acc}')

                results.setdefault(run_root_template, []).append(eval_list)

        json.dump(results, open(f'{decode_file_name}.eval_results.json', 'w'), indent=2)
