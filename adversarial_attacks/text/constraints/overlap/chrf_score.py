import nltk.translate.chrf_score

from ..constraint import Constraint


class chrF(Constraint):

    def __init__(self, max_chrf, compare_against_original=True):
        super().__init__(compare_against_original)
        if not isinstance(max_chrf, int):
            raise TypeError("max_chrf must be an int")
        self.max_chrf = max_chrf

    def _check_constraint(self, transformed_text, reference_text):
        ref = reference_text.words
        hyp = transformed_text.words
        chrf = nltk.translate.chrf_score.sentence_chrf(ref, hyp)
        return chrf <= self.max_chrf
