import torch
import bert_score

from ..constraint import Constraint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTScore(Constraint):

    SCORE_TYPE2IDX = {"precision": 0, "recall": 1, "f1": 2}

    def __init__(
        self,
        min_bert_score,
        model_name="bert-base-uncased",
        num_layers=None,
        score_type="f1",
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        if not isinstance(min_bert_score, float):
            raise TypeError("max_bert_score must be a float")
        if min_bert_score < 0.0 or min_bert_score > 1.0:
            raise ValueError("max_bert_score must be a value between 0.0 and 1.0")

        self.min_bert_score = min_bert_score
        self.model = model_name
        self.score_type = score_type
        # Turn off idf-weighting scheme b/c reference sentence set is small
        self._bert_scorer = bert_score.BERTScorer(
            model_type=model_name, idf=False, device=device, num_layers=num_layers
        )

    def _sim_score(self, starting_text, transformed_text):
        cand = transformed_text.text
        ref = starting_text.text
        result = self._bert_scorer.score([cand], [ref])
        return result[BERTScore.SCORE_TYPE2IDX[self.score_type]].item()

    def _check_constraint(self, transformed_text, reference_text):

        score = self._sim_score(reference_text, transformed_text)
        if score >= self.min_bert_score:
            return True
        else:
            return False