"""
Generate data from the official calflow data release

Usage:
    preprocess_dataset.py --data-path=<path> --output-path=<path>  --seed=<int>
"""


import re
from pathlib import Path
import json
from typing import List, Tuple, Dict

from dataflow.core.linearize import lispress_to_seq
from tqdm import tqdm
import numpy as np
from docopt import docopt

from dataflow.core.sexp import Sexp, flatten, sexp_to_str
from dataflow.core.lispress import parse_lispress

from utils.sexp_utils import sexp_to_string


def exact(s):
    return lambda tok: tok == s


def contains(s):
    return lambda tok: s in tok


TRIGGERS = {
    "all": {lambda tok: True},
    "weather": {exact("DarkSkyQueryApi"), contains("WeatherProp")},
    "orgchart": {exact("FindManager"), exact("FindReports"), exact("FindTeamOf")},
    "places": {exact("FindPlace"), exact("PlaceHasFeature"), exact("Here")},
    "events": {contains("Event")},
    "fence": {contains("Fence"), contains("Pleasantry")},
}


def tags_for(plan):
    if isinstance(plan, str):
        plan = plan.split()

    out = set()
    for tok in plan:
        for tag, triggers in TRIGGERS.items():
            for trigger in triggers:
                if trigger(tok):
                    out.add(tag)

    return set(out)


def get_example_tags(utterance: str, plan: str) -> Tuple[str]:
    tags = tags_for(plan)

    if 'events' not in tags and 'orgchart' in tags:
        tags.add('simple_orgchart')

    if 'events' in tags and 'orgchart' in tags:
        tags.add('events_with_orgchart')

    if 'events' in tags and 'orgchart' not in tags:
        tags.add('events_without_orgchart')

    if 'events' in tags and 'weather' not in tags:
        tags.add('events_without_weather')

    if 'events' in tags and 'weather' in tags:
        tags.add('events_with_weather')

    if 'events' not in tags and 'weather' in tags:
        tags.add('simple_weather')

    if 'events' in tags and 'weather' not in tags:
        tags.add('events_without_weather')

    if 'events' in tags:
        if 'DeleteCommitEventWrapper' in plan:
            tags.add('events.delete')
        elif 'UpdateCommitEventWrapper' in plan:
            tags.add('events.update')
        elif 'CreateCommitEventWrapper' in plan:
            tags.add('events.create')
        elif 'FindEventWrapperWithDefaults' in plan:
            tags.add('events.query')

    return tags


PERSON_NAME_RE = re.compile(
    r"\(refer \(extensionConstraint \(RecipientWithNameLike :constraint \(Constraint\[Recipient\]\) :name # \(PersonName \"[a-zA-Z ]+?\"\)\)\)\)"
)


def structure_match(root: Sexp) -> bool:
    if isinstance(root, str):
        return True

    op_name = (
        root[0]
        if isinstance(root, list) and isinstance(root[0], str)
        else None
    )

    if op_name == 'refer':  # GetSalient
        # check if the structure is similar to
        # (refer
        #   (extensionConstraint
        #       (RecipientWithNameLike
        #           :constraint (Constraint[Recipient])
        #           :name #(PersonName \"Marton\")
        #       )
        #   )
        # )

        node_string_expr = sexp_to_str(root)
        if PERSON_NAME_RE.match(node_string_expr):
            return True
        else:
            return False
    else:
        match_result = True
        for child_node in root:
            match_result &= structure_match(child_node)

        return match_result


def inline_variable(root: Sexp) -> Sexp:
    variables = {}

    def _find_variable_definition(_root: Sexp):
        if isinstance(_root, list) and len(_root) > 1 and _root[0] == 'let':
            var_def = _root[1]
            ptr = 0
            while ptr < len(var_def):
                assert re.match(r'x\d', var_def[ptr])
                var_id = var_def[ptr]
                var_body = var_def[ptr + 1]
                ptr += 1

                if var_body == '#':
                    var_body = [var_body, var_def[ptr + 1]]
                    ptr += 1

                variables[var_id] = var_body
                ptr += 1

            # var_id = var_def[0]
            # var_body = var_def[1]
            expr = _root[2]
            # var_body = _root[2]
            # var_body_string = sexp_to_string(var_body)
            # if (
            #
            #     # var_body_string.startswith("""( Execute :intension ( GetSalient :constraint""") and
            #     # 'RecipientWithNameLike' in var_body_string
            # ):
        elif isinstance(_root, list):
            for child in _root:
                _find_variable_definition(child)

    def _replace(_root: Sexp) -> Sexp:
        # unpack the root let node
        if isinstance(_root, list) and len(_root) > 1 and _root[0] == 'let':
            var_def = _root[1]
            expr = _root[2]

            return _replace(expr)
        else:
            if isinstance(_root, str):
                if _root in variables:
                    return variables[_root]

                return _root

            return [
                _replace(child)
                for child in _root
            ]

    _find_variable_definition(root)

    if variables:
        return _replace(root)
    else:
        return root


def canonicalize_program_tokens(plan_tokens: List[str]) -> (List[str], List[int]):
    canonical_tokens = []
    canonical_token_index_map = []
    for original_idx, token in enumerate(plan_tokens):
        if (
            token not in {'(', ')', '#', '"'}
            and not token.startswith("'?")
            # and not t.startswith(":")
        ):
            canonical_tokens.append(token)
            canonical_token_index_map.append(original_idx)

    return canonical_tokens, canonical_token_index_map


def canonicalize_program(plan: str) -> str:
    return ' '.join(
        canonicalize_program_tokens(plan.split(' '))[0]
    )


def canonicalize_examples(examples: List[Dict]) -> List[Dict]:
    canonicalized_examples = []
    for example in examples:
        new_example = dict(example)
        new_example['plan'] = canonicalize_program(example['plan'])

        canonicalized_examples.append(new_example)

    return canonicalized_examples


def extract_examples(file_path: Path):
    dialogues = [
        json.loads(line)
        for line in file_path.open()
    ]
    examples = []

    for line_idx, dialogue in enumerate(tqdm(dialogues), start=1):
        for turn in dialogue['turns']:
            turn_idx = turn['turn_index']
            program_string = turn['lispress']

            if (
                any(
                    x in program_string
                    for x
                    in [
                        'ReviseConstraint',
                        'Pleasantry',
                        'DoNotConfirm',
                        'Fence',
                        'ConfirmAndReturnAction',
                        'ExternalReference',   # TODO: this is not in the new data
                        'DoNotConfirm',
                        'UserPauseResponse',
                        'DateTimeAndConstraintBetweenEvents'  # TODO: added by Pengcheng
                    ]
                ) or
                program_string == '(Yield :output (CurrentUser))'
            ):
                continue

            current_user_utterance = turn['user_utterance']['original_text']
            utterance_lower = current_user_utterance.lower()

            if (
                current_user_utterance == '__NULL' or
                'does it' in utterance_lower or
                'make it' in utterance_lower
            ):
                continue

            target_sexp: Sexp = parse_lispress(program_string)
            target_sexp_tokens: List[str] = lispress_to_seq(target_sexp)
            target_sexp_token_num = len(target_sexp_tokens)
            utterance_token_num = len(turn['user_utterance']['tokens'])

            if target_sexp_token_num / utterance_token_num > 25 or target_sexp_token_num > 250 or utterance_token_num <= 1:
                print(f'Pruned utterance:s {current_user_utterance}')
                continue

            if turn_idx > 0:
                # we retain certain GetSalient calls.
                is_structure_match = structure_match(target_sexp)
                if not is_structure_match:
                    continue

            variable_inlined_sexp = inline_variable(target_sexp)
            variable_inlined_sexp_tokens = lispress_to_seq(variable_inlined_sexp)
            target_sexp_string = ' '.join(variable_inlined_sexp_tokens)

            if len(variable_inlined_sexp_tokens) > 250:
                continue

            # if 'FindManager' in tgt_line and 'attendees' not in tgt_line:
            # if line_idx < 20:
            #     print(f'{line_idx}: {current_user_utterance}\t{target_sexp_string}')

            tags = sorted(get_example_tags(current_user_utterance, target_sexp_string))

            canonical_tokens, canonical_token_map = canonicalize_program_tokens(variable_inlined_sexp_tokens)

            if line_idx < 3000:
                for token, original_idx in zip(canonical_tokens, canonical_token_map):
                    assert variable_inlined_sexp_tokens[original_idx] == token

            examples.append({
                'dialogue_id': dialogue['dialogue_id'],
                'turn_index': turn_idx,
                'utterance': ' '.join(turn['user_utterance']['tokens']),
                'plan': target_sexp_string,
                'canonical_tokens': canonical_tokens,
                'canonical_token_index_map': canonical_token_map,
                'tags': tags,
            })

    return examples


def generate_splits(examples: List[Dict], seed: int):
    def has_tag(tag):
        return lambda _example: tag in _example['tags']

    splits = {
        # 'orgchart': {
        #     'source_domains': ['simple_orgchart', 'events_without_orgchart'],
        #     'target_domains': ['events_with_orgchart']
        # },
        'orgchart.event_create': {
            'source_domains': [['simple_orgchart'], ['events_without_orgchart', 'events.create']],
            'target_domains': [['events_with_orgchart', 'events.create']]
        },
    }

    all_splits = {}
    for split_name, domains in splits.items():
        # seed = 199201
        np.random.seed(seed)

        domain_data = {}
        for domain, tags in domains.items():
            for example in examples:
                if any(
                    (
                        has_tag(tag)(example)
                        if isinstance(tag, str)
                        else all(
                            has_tag(tag_i)(example) for tag_i in tag
                        )
                    )
                    for tag
                    in tags
                ):
                    domain_data.setdefault(domain, []).append(example)

        # add some hard examples from target domain to source domains
        target_domain_data = domain_data['target_domains'][:]

        num_target_examples_in_train = [
            0, 8, 16,
            32, 64,
            128,
            # 256,
        ]

        # we make sure the sub-sampled training set should contain the following key phrases.
        constrained_tokens = [
            "my skip",
            "'s skip",
            "'s team",
            "'s boss",
            "'s manager",
            "my reports",
            "his reports",
            "'s reports",
            "his team",
            "his boss",
            "his manager",
            "her boss",
            "her manager",
            "her team",
            "my boss",
            "my team",
            "my manager",
            "supervisor",
        ]

        seed_i = seed
        satisfied = False

        num_reserved_target_domain_examples_for_val_test = len(target_domain_data) - max(num_target_examples_in_train)

        while not satisfied:
            np.random.shuffle(target_domain_data)

            satisfied = all(
                any(
                    seq in e["utterance"]
                    for e
                    in target_domain_data[num_reserved_target_domain_examples_for_val_test:num_reserved_target_domain_examples_for_val_test + 32]
                )
                for seq
                in constrained_tokens
            )

            seed_i += 1

        reserved_target_examples_for_val_test = target_domain_data[:num_reserved_target_domain_examples_for_val_test]

        print(f'split: {split_name}, num. target evaluation examples: {len(reserved_target_examples_for_val_test)}')

        reserved_source_domain_test_examples_num = num_reserved_target_domain_examples_for_val_test // 2
        reserved_source_domain_val_examples_num = reserved_source_domain_test_examples_num

        assert len(domain_data['source_domains']) - reserved_source_domain_test_examples_num > reserved_source_domain_test_examples_num

        indices = list(range(len(domain_data['source_domains'])))
        np.random.shuffle(indices)

        source_domain_train_indices = indices[reserved_source_domain_val_examples_num + reserved_source_domain_test_examples_num:]
        source_domain_val_indices = indices[:reserved_source_domain_val_examples_num]
        source_domain_test_indices = indices[
             reserved_source_domain_val_examples_num:
             reserved_source_domain_val_examples_num + reserved_source_domain_test_examples_num
        ]

        assert len(source_domain_train_indices) == len(set(source_domain_train_indices))
        assert len(source_domain_val_indices) == len(set(source_domain_val_indices))
        assert len(source_domain_test_indices) == len(set(source_domain_test_indices))

        source_domains_train_data = [
            domain_data['source_domains'][idx]
            for idx
            in source_domain_train_indices
        ]

        source_domains_val_data = [
            domain_data['source_domains'][idx]
            for idx
            in source_domain_val_indices
        ]

        source_domains_test_data = [
            domain_data['source_domains'][idx]
            for idx
            in source_domain_test_indices
        ]

        target_domain_val_data = reserved_target_examples_for_val_test[:reserved_source_domain_val_examples_num]
        target_domain_test_data = reserved_target_examples_for_val_test[reserved_source_domain_val_examples_num:]

        # domain_data['source_domains'] = source_domains_train_data

        candidate_target_domain_examples_for_training = target_domain_data[num_reserved_target_domain_examples_for_val_test:]
        # rng = np.random.RandomState(seed=1234)
        # rng.shuffle(candidate_target_domain_examples_for_training)

        all_splits[split_name] = {}
        for tgt_example_num in num_target_examples_in_train:
            assert len(candidate_target_domain_examples_for_training) >= tgt_example_num

            target_examples_to_include = candidate_target_domain_examples_for_training[:tgt_example_num]
            train_set = source_domains_train_data + target_examples_to_include

            all_splits[split_name][f'source_domain_with_target_num{tgt_example_num}'] = train_set

            print(f'Generated train set of size {len(train_set)}, with {tgt_example_num} examples from tgt domain.')

        # all_splits[split_name]['target_domains'] = reserved_target_examples_for_val_test
        dev_set = target_domain_val_data + source_domains_val_data
        test_set = target_domain_test_data + source_domains_test_data

        # assert not set(dev_set).intersection(test_set)

        all_splits[split_name]['dev_set'] = dev_set
        all_splits[split_name]['test_set'] = test_set

        print(f'Generated dev set of size {len(dev_set)}, '
              f'with {len(source_domains_val_data)} examples from src domain, '
              f'and {len(target_domain_val_data)} examples from tgt domain.')
        print(f'Generated dev set of size {len(test_set)}, '
              f'with {len(source_domains_test_data)} examples from src domain, '
              f'and {len(target_domain_test_data)} examples from tgt domain.')

    return all_splits


def output_examples_jsonl(examples: List[Dict], file_path: str):
    file_path = Path(file_path)

    onmt_src_file = file_path.with_suffix('.src')
    onmt_tgt_file = file_path.with_suffix('.tgt')

    # datum_id_file = file_path.with_suffix('.datum_id')

    file_path.parent.mkdir(exist_ok=True, parents=True)

    with file_path.open('w') as f, onmt_src_file.open('w') as f_src, onmt_tgt_file.open('w') as f_tgt:
        for example in examples:
            f.write(json.dumps(example) + '\n')

            f_src.write(example['utterance'] + '\n')
            f_tgt.write(example['plan'] + '\n')
            # f_datum_id.write(example['datum_id'] + '\n')


def generate_data(data_root: Path, output_root: Path, seed: int=199201):
    train_examples = extract_examples(data_root / 'train.dataflow_dialogues.jsonl')

    valid_examples = extract_examples(data_root / 'valid.dataflow_dialogues.jsonl')

    output_root.resolve().mkdir(parents=True, exist_ok=True)

    all_splits = generate_splits(train_examples + valid_examples, seed=seed)

    for domain_name, domain_splits in all_splits.items():
        domain_file_prefix = f'calflow.{domain_name}'

        dev_set = domain_splits['dev_set']
        test_set = domain_splits['test_set']

        train_data_dict = {
            k: v
            for k, v
            in domain_splits.items()
            if k.startswith('source_domain_with_target_num')
        }

        for split_name, split in train_data_dict.items():
            split_file = output_root / domain_file_prefix / split_name / 'train.jsonl'

            output_examples_jsonl(split, split_file)
            output_examples_jsonl(dev_set, split_file.with_name('valid.jsonl'))
            output_examples_jsonl(test_set, split_file.with_name('test.jsonl'))

            # output canonical version of the data
            canonical_split = canonicalize_examples(split)
            canonical_valid = canonicalize_examples(dev_set)
            canonical_test = canonicalize_examples(test_set)

            output_examples_jsonl(canonical_split, split_file.with_suffix('.canonical.jsonl'))
            output_examples_jsonl(canonical_valid, split_file.with_name('valid.canonical.jsonl'))
            output_examples_jsonl(canonical_test, split_file.with_name('test.canonical.jsonl'))


if __name__ == '__main__':
    args = docopt(__doc__)

    generate_data(
        data_root=Path(args['--data-path']),
        output_root=Path(args['--output-path']),
        seed=int(args['--seed'])
    )
