# alignment for the spider dataset
import json
from pathlib import Path
from typing import Dict, List

from utils.sql.spider_utils.process_sql import tokenize


def preprocess_dataset(file_path: Path, delexicalized: bool = True):
    examples_dict: List[Dict] = json.load(file_path.open())

    for example_idx, example in enumerate(examples_dict):
        src_tokens = example['question_toks']
        sql_query = example['query']
        sql_query_tokens = tokenize(sql_query)

        print(example['question'])
        print(sql_query)
        print()


if __name__ == '__main__':
    preprocess_dataset(Path('~/Research/datasets/spider/train_spider.json').expanduser())
