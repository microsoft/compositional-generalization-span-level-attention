from collections import defaultdict
from typing import Iterable, Any, Dict, List, Set, TypeVar, Type
from overrides import overrides

from allennlp.training.metrics import Metric


T = TypeVar('T')


@Metric.register('categorical')
class CategoricalMetric(Metric):
    def __init__(self, metric_class: Type[Metric]):
        self._metric_cls = metric_class
        self._named_metrics = defaultdict(metric_class)
        self.reset()

    def __call__(self, hypotheses: List[T], references: List[T], *args, tags: List[List[str]], **kwargs):
        for example_id, (hypothesis, reference, example_tags) in enumerate(zip(hypotheses, references, tags)):
            for tag in example_tags:
                self._named_metrics[tag](
                    [hypothesis],
                    [reference],
                    *[[arg[example_id]] for arg in args],
                    **{arg_name: [arg_val[example_id]] for arg_name, arg_val in kwargs.items()}
                )

    def get_metric(
        self, reset: bool
    ) -> Dict[str, float]:
        all_metrics = {}
        for tag_name, metric in self._named_metrics.items():
            metric_dict = metric.get_metric(reset)
            all_metrics.update({
                f'{tag_name}_{name}': val
                for name, val in metric_dict.items()
            })

        return all_metrics

    def reset(self) -> None:
        self._named_metrics.clear()
