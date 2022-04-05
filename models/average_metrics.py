from typing import Optional

from overrides import overrides

from allennlp.training.metrics import Average
from allennlp.training.metrics.metric import Metric


@Metric.register("average_meter")
class AverageMeter(Average):
    """A metric that resets itself by default"""
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def get_metric(self, reset: bool = True) -> Optional[float]:
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """
        average_value = self._total_value / self._count if self._count > 0 else None
        self.reset()

        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
