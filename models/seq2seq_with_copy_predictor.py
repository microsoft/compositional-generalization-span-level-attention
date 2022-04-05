from allennlp.common import JsonDict
from allennlp.data import Instance
from overrides import overrides

from allennlp.predictors import Predictor


@Predictor.register('seq2seq_copy')
class Seq2SeqModelWithCopyPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)
