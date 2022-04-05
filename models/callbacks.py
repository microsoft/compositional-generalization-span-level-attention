from typing import Dict, Any
from allennlp.training import EpochCallback, Trainer

try:
    import wandb

    @EpochCallback.register('wandb')
    class WandbLoggerEpochCallBack(EpochCallback):
        def __call__(
            self,
            trainer: Trainer,
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
        ) -> None:
            if wandb.run is not None and epoch >= 0:
                wandb.log({'epoch': epoch, **metrics})
except:
    pass
