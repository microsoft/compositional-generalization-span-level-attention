from collections import defaultdict
from typing import Iterable, Any, Dict, List, Set, TypeVar
from overrides import overrides

from allennlp.training.metrics import Metric


T = TypeVar('T')


@Metric.register('oracle')
class OracleMetric(Metric):
    def __init__(self, metric_instance: Metric):
        self._metric = metric_instance
        self._counters = defaultdict(int)
        self.reset()

    def __call__(self, hypotheses: List[List[T]], reference: List[T], **kwargs):
        for i, (hyp_list, reference) in enumerate(zip(hypotheses, reference)):
            hyp_metrics = []

            for hyp_id, hyp in enumerate(hyp_list):
                self._metric(
                    [hyp], [reference],
                    **{k: [v[i]] for k, v in kwargs.items()}
                )
                hyp_metric = self._metric.get_metric().copy()

                hyp_metrics.append(hyp_metric)
                self._metric.reset()

            all_keys = {
                key
                for metric in hyp_metrics
                for key in metric
                if key.endswith('_em')
            }

            for key in all_keys:
                if any(
                    metric.get(key) for metric in hyp_metrics
                ):
                    self._counters[f'oracle_{key}'] += 1
                else:
                    self._counters[f'oracle_{key}'] += 0

                self._counters[f'oracle_{key}_count'] += 1

    def get_metric(
        self, reset: bool
    ) -> Dict[str, float]:
        metric = {}
        for key in [
            key
            for key in self._counters
            if key.endswith('_em')
        ]:
            count = self._counters[f'{key}_count']
            metric[key] = 0. if count == 0 else self._counters[key] / count

        return metric

    def reset(self) -> None:
        self._counters.clear()
