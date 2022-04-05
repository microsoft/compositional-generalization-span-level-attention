from pathlib import Path
from typing import List, Dict, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np

from models.seq2seq_parser_reader import SequenceToSequenceModelWithCopyReader


def find_substree_span_at(tgt_tokens, start_idx):
    stack = []
    pushed = False
    idx = start_idx
    while idx < len(tgt_tokens):
        token = tgt_tokens[idx]
        if token == '(':
            pushed = True
            stack.append(token)
        elif token == ')':
            assert len(stack) > 0
            stack.pop()

        idx += 1
        if len(stack) == 0 and pushed:
            return (start_idx, idx)

    raise IndexError()


def get_subtree_tokens_and_spans_for_arg(tgt_tokens, arg_name):
    t_subtree_start = tgt_tokens.index(arg_name)
    subtree_span = find_substree_span_at(tgt_tokens, t_subtree_start)
    hyp_subtree_tokens: List[str] = tgt_tokens[subtree_span[0]: subtree_span[1]]

    return hyp_subtree_tokens, subtree_span


def get_attention_quality(decoder_attn_distribution: np.ndarray, oracle_src_span: Tuple[int]) -> float:
    src_span_start, src_span_end = oracle_src_span

    q = decoder_attn_distribution[:, src_span_start: src_span_end].sum(axis=-1).mean()
    # uniform = np.zeros((decoder_attn_distribution.shape[0], decoder_attn_distribution.shape[1]))
    # uniform[:, src_span_start: src_span_end] = 1. / (src_span_end - src_span_start)
    # q = ((decoder_attn_distribution - uniform) ** 2).mean()

    return float(q)


def analyze_example(decode_result_dict: Dict, arg_name: str):
    sys_decode_results_file_path = Path()
    src_tokens = decode_result_dict['metadata']['source_tokens']
    src_subtoken_offsets = decode_result_dict['metadata']['source_subtoken_offsets']
    tgt_tokens = decode_result_dict['metadata']['target_tokens']
    oracle_alignment_info = decode_result_dict['metadata']['alignment_info']

    # (target_decode_step, source_length)
    decoder_attn_weights: np.ndarray = np.array(decode_result_dict['best_prediction_attention_weights'])

    hyp_tokens = decode_result_dict['predictions'][0]['tokens']
    # = ':attendees'

    has_desired_substree = arg_name in hyp_tokens

    output = {
        'is_valid_example': False
    }

    if arg_name not in tgt_tokens:
        return output

    if has_desired_substree:
        hyp_subtree_tokens, (t_subtree_start, t_subtree_end) = get_subtree_tokens_and_spans_for_arg(hyp_tokens, arg_name)
        tgt_subtree_tokens, tgt_subtree_span = get_subtree_tokens_and_spans_for_arg(tgt_tokens, arg_name)

        # get alignment entries for this source span
        subtree_oracle_alignments = [
            alignment_entry
            for alignment_entry
            in oracle_alignment_info
            if alignment_entry['target_tokens'][0] == arg_name
        ]
        if len(subtree_oracle_alignments) < 1:
            return output

        subtree_oracle_alignment: Dict = subtree_oracle_alignments[0]

        oracle_aligned_src_tokens, (oracle_src_span_start, oracle_src_span_end) = SequenceToSequenceModelWithCopyReader._get_subtokens_slice(
            src_tokens,
            subtree_oracle_alignment['source_tokens_idx'],
            src_subtoken_offsets
        )

        # oracle_subtree_alignment_distribution = oracle_alignment_distribution[tgt_subtree_span[0]: tgt_subtree_span[1]]
        # num source tokens
        # p_oracle_subtree_align = tgt_subtree_span[0]

        decoder_subtree_attn_weights: np.ndarray = decoder_attn_weights[t_subtree_start: t_subtree_end]
        attn_quality: float = get_attention_quality(
            decoder_subtree_attn_weights,
            (oracle_src_span_start, oracle_src_span_end)
        )

        is_subtree_correct = hyp_subtree_tokens == tgt_subtree_tokens

        output.update({
            'is_valid_example': True,
            'attn_quality': attn_quality,
            'is_subtree_correct': is_subtree_correct
        })

    return output


def main():
    # decode_file_path = Path('data/remote_runs/model.all.ber2seq.span_attn.tgt32.pred.valid.jsonl')
    # decode_file_path = Path('data/remote_runs/model.all.ber2seq.token_attn.tgt32.pred.valid.jsonl')
    decode_file_path = Path('data/remote_runs/run_210409_204453388182_seq2seq_bert_run2_seed2/valid.alignment.5a908cdf.decode.with_attn_score.jsonl')

    decode_results = [
        json.loads(line)
        for line
        in decode_file_path.open()
    ]

    examples_we_care = [
        example_decode_result
        for example_decode_result
        in decode_results
        if 'events_with_orgchart' in example_decode_result['metadata']['tags']
    ]

    acc = np.average(
        np.array(
            [
                True if x['predictions'][0]['tokens'] == x['metadata']['target_tokens'] else False
                for x in examples_we_care
            ]
        )
    )

    print(f'Avg. Acc={acc}')

    print(len(examples_we_care))

    results = []
    for example_decode_result in examples_we_care:
        for arg_name in [':attendees']:  # , ':subject', ':start', ':location'
            example_analyze_result = analyze_example(example_decode_result, arg_name)
            if example_analyze_result['is_valid_example']:
                results.append(example_analyze_result)

    correct_samples_attn_metric = np.array([
        x['attn_quality']
        for x in results
        if x['is_subtree_correct']
    ])

    # plt.hist(correct_samples_attn_metric)
    # plt.title('Hist of correct samples')
    # plt.show()

    incorrect_samples_attn_metric = np.array([
        x['attn_quality']
        for x in results
        if not x['is_subtree_correct']
    ])

    # plt.hist(incorrect_samples_attn_metric)
    # plt.title('Hist of incorrect samples')
    # plt.show()

    # print(np.average(correct_samples_attn_metric))
    # print(np.average(incorrect_samples_attn_metric))

    # scatter plot
    xs = []
    ys = []

    for entry in results:
        x = entry['attn_quality']
        y = float(entry['is_subtree_correct'])

        xs.append(x)
        ys.append(y)

    plt.scatter(xs, ys)
    plt.show()

    step_size = 10
    high = 100
    xs = []
    x_labels = []
    ys = []
    while True:
        low = high - step_size
        if high == 40:
            low = 0

        result_list_with_attn_quality_bucket = np.array([
            x['is_subtree_correct']
            for x in results
            if low / 100 < x['attn_quality'] <= high / 100
        ])

        x = len(x_labels)
        x_label = f'({low}, {high}]'
        y = np.average(result_list_with_attn_quality_bucket)
        print(f'{x_label}: {y} ({len(result_list_with_attn_quality_bucket)} samples)')

        xs.append(x)
        x_labels.append(x_label)
        ys.append(y)

        high = low

        if high == 0:
            break

    plt.plot(xs, ys)
    plt.xticks(xs, x_labels)
    plt.title('Sub-program acc. w.r.t the averaged attention score over oracle utterance spans.')
    plt.show()

    with open(decode_file_path.with_suffix('.attn_score_wrt_acc.json'), 'w') as f:
        f.write(
            json.dumps({
                'x': xs,
                'x_labels': x_labels,
                'ys': [float(y) for y in ys]
            })
        )

    result_list_high_attn_quality = np.array([
        x['is_subtree_correct']
        for x in results
        if x['attn_quality'] >= 0.9
    ])

    result_list_low_attn_quality = np.array([
        x['is_subtree_correct']
        for x in results
        if x['attn_quality'] < 0.9
    ])

    print(np.average(result_list_high_attn_quality))
    print(np.average(result_list_low_attn_quality))


if __name__ == '__main__':
    main()
