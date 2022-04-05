from collections import defaultdict
from typing import Iterable, Any, Dict, List, Set
from overrides import overrides

from allennlp.training.metrics import Metric


@Metric.register('sequence_match')
class SequenceMatchingMetric(Metric):
    def __init__(self) -> None:
        self.reset()

    @staticmethod
    def f1(predicted: Iterable[Any], expected: Iterable[Any]) -> float:
        expected = frozenset(expected)
        predicted = frozenset(predicted)
        if len(predicted) <= 0 and len(expected) <= 0:
            return 1.0
        if len(predicted) <= 0 or len(expected) <= 0:
            return 0.0

        true_positive_count = len(predicted & expected)
        p = true_positive_count / len(predicted)
        r = true_positive_count / len(expected)
        return (2 * p * r) / (p + r)

    @overrides
    def __call__(self, best_span_strings, answer_strings):
        for best_span_string, answer_string in zip(best_span_strings, answer_strings):
            self._total_em += best_span_string == answer_string
            self._total_f1 += self.f1(best_span_string, answer_string)
            self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return {"em": exact_match, "f1": f1_score}

    @overrides
    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"SequenceMatchingMetric(em={self._total_em}, f1={self._total_f1})"


@Metric.register('sequence_categorized_match')
class SequenceCategorizedMatchMetric(Metric):
    def __init__(self) -> None:
        self._counters = {}
        self.reset()

    @overrides
    def __call__(self, best_span_strings: List[Any], answer_strings: List[Any], tag_set_lists: List[Iterable[str]] = None):
        processed_tag_set_lists = []

        tag_set_lists = tag_set_lists or [{'all'}] * len(best_span_strings)
        for tags in tag_set_lists:
            if not tags:
                tags = {'all'}
            processed_tag_set_lists.append(tags)
        tag_set_lists = processed_tag_set_lists

        for best_span_string, answer_string, tags in zip(best_span_strings, answer_strings, tag_set_lists):
            em = best_span_string == answer_string

            for tag in tags:
                self._counters.setdefault(tag, defaultdict(int))['em'] += em
                self._counters.setdefault(tag, defaultdict(int))['count'] += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}

        for tag, metric in self._counters.items():
            exact_match = metric['em'] / metric['count'] if metric['count'] > 0 else 0.
            metrics.update({
                f'{tag}_em': exact_match,
                f'{tag}_count': metric['count']
            })

        if reset:
            self.reset()

        return metrics

    @overrides
    def reset(self):
        self._counters = {}

    def __str__(self):
        return f"SequenceCategorizedMatchMetric({self.get_metric()})"
