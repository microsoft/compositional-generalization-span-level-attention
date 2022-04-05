import json
from pathlib import Path
from typing import List, Tuple

from utils.train_giza_alignment_general import train_giza_bidirectional
from utils.sql.sql_utils.text2sql_utils import SqlTokenizer, canonicalize_sql_for_alignment, process_sql_data_standard, CANONICAL_VARIABLES
from utils.sql.sql_utils.encoder_input_canonicalizer import process_sentence


# add ATIS alignment rules
atis_alignment_rules = [(var, var) for var in sorted(CANONICAL_VARIABLES)]

# the below heuristic alignments are NOT used in alignment extraction.
atis_alignment_rules.extend([
    (
        ['flights', 'flight'],
        ['FLIGHTalias0.FLIGHT_ID']
    ),
    (
        ['airline', 'airlines'],
        ['AIRLINEalias0.AIRLINE_CODE', 'AIRLINEalias1.AIRLINE_CODE']
    ),
    (
        ['flight number', 'flight numbers'],
        ['FLIGHTalias0.FLIGHT_NUMBER', 'FLIGHTalias1.FLIGHT_NUMBER']
    ),
    (
        ['from city_name0', 'leave city_name0', 'leaving city_name0', 'departing from city_name0', 'leaving from city_name0'],
        [
            "CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = city_name0 AND FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE",
            "CITYalias1.CITY_CODE = AIRPORT_SERVICEalias1.CITY_CODE AND CITYalias1.CITY_NAME = city_name0 AND FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE",
        ]
    ),
    (
        ['from city_name1', 'leave city_name1', 'leaving city_name1', 'departing from city_name1', 'leaving from city_name1'],
        [
            "CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = city_name1 AND FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE",
            "CITYalias1.CITY_CODE = AIRPORT_SERVICEalias1.CITY_CODE AND CITYalias1.CITY_NAME = city_name1 AND FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE",
        ]

    ),
    (
        ['to city_name0', 'arriving in city_name0', 'returning to city_name0'],
        [
            "CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = city_name0 AND FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE",
            "CITYalias1.CITY_CODE = AIRPORT_SERVICEalias1.CITY_CODE AND CITYalias1.CITY_NAME = city_name0 AND FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE",
        ]
    ),
    (
        ['to city_name1', 'arriving in city_name1', 'returning to city_name1'],
        [
            "CITYalias0.CITY_CODE = AIRPORT_SERVICEalias0.CITY_CODE AND CITYalias0.CITY_NAME = city_name1 AND FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE",
            "CITYalias1.CITY_CODE = AIRPORT_SERVICEalias1.CITY_CODE AND CITYalias1.CITY_NAME = city_name1 AND FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE",
        ]
    ),
    (
        ['city_name0'],
        [
            "CITYalias0.CITY_NAME = city_name0",
            "CITYalias1.CITY_NAME = city_name0"
        ]
    ),
    (
        ['city_name1'],
        [
            "CITYalias0.CITY_NAME = city_name0",
            "CITYalias1.CITY_NAME = city_name0"
        ]
    ),
    # (
    #     ['from', 'leave', 'leaving', 'departing from', 'leaving from'],
    #     [
    #         "FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE",
    #         "FLIGHTalias0.FROM_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE"
    #     ]
    # ),
    # (
    #     ['to', 'arriving in', 'returning'],
    #     [
    #         "FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias0.AIRPORT_CODE",
    #         "FLIGHTalias0.TO_AIRPORT = AIRPORT_SERVICEalias1.AIRPORT_CODE"
    #     ]
    # ),
    (
        ['afternoon flight', 'afternoon', 'morning flight', 'morning', 'evening', 'evening flight'],
        "FLIGHTalias0.DEPARTURE_TIME BETWEEN departure_time0 AND departure_time1"
    ),
    (
        ['1 way'],
        [
            "FAREalias0.ROUND_TRIP_REQUIRED = round_trip_required0",
            "FAREalias1.ROUND_TRIP_REQUIRED = round_trip_required0"
        ]
    )
])

domain_specific_alignment_rules = {
    'atis': atis_alignment_rules
}


def ensure_list(items):
    if not isinstance(items, list):
        items = [items]

    return items


def add_domain_specific_rule(target_list: List, rule: Tuple):
    for src in ensure_list(rule[0]):
        for tgt in ensure_list(rule[1]):
            target_list.append((src, tgt))

    return target_list


def run_alignment(dataset_path: Path, split: str = None, domain_rule_repeat_time: int = 1000):
    dataset = json.load(dataset_path.open())
    sql_tokenizer = SqlTokenizer()

    raw_examples: List[Tuple[str, str]] = list(process_sql_data_standard(
        dataset,
        use_linked=True,
        use_all_queries=True,
        use_all_sql=False
    ))

    num_raw_examples = len(raw_examples)
    if split:
        domain_alignment_rules = domain_specific_alignment_rules.get(split, [])
        for rule in domain_alignment_rules:
            for t in range(domain_rule_repeat_time):
                add_domain_specific_rule(raw_examples, rule)

    examples = []
    for idx, (source, sql_query) in enumerate(raw_examples):
        sql_tokens = sql_tokenizer.tokenize(sql_query)
        if idx < num_raw_examples:
            canonical_sql_tokens, sql_token_offsets = canonicalize_sql_for_alignment(sql_tokens)
            canonical_sql_query = ' '.join(canonical_sql_tokens)
        else:
            canonical_sql_query = sql_query
            sql_token_offsets = None

        canonical_source = process_sentence(source)
        source_tokens = source.split()

        examples.append((canonical_source, canonical_sql_query, sql_token_offsets))

    src_file = dataset_path.with_suffix('.src')
    tgt_file = dataset_path.with_suffix('.tgt')
    meta_file = dataset_path.with_suffix('.alignment_meta.jsonl')
    with src_file.open('w') as f_src, tgt_file.open('w') as f_tgt, meta_file.open('w') as f_meta:
        for example in examples:
            source, target, sql_token_offsets = example
            f_src.write(source + '\n')
            f_tgt.write(target + '\n')
            f_meta.write(json.dumps({
                'simplified_sql_for_alignment_to_original_token_offset': sql_token_offsets
            }) + '\n')

    train_giza_bidirectional(
        src_file,
        tgt_file,
        label=f'a0a017e1_rulerpt{domain_rule_repeat_time}'
    )


def main():
    dataset_root = Path('/path/to/improving-compgen-in-semparse/data/sql_data/')

    domain_alignment_rule_repeat_time = 0
    for dataset in ['atis']:  # 'atis' 'advising', 'scholar'
        for split in ['new_question_split', 'schema_full_split']:
            dataset_path = dataset_root / dataset / split
            run_alignment(
                dataset_path / 'aligned_train.json',
                split=dataset,
                domain_rule_repeat_time=domain_alignment_rule_repeat_time
            )


if __name__ == '__main__':
    main()
