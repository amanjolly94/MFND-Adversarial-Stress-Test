import nltk

from ..constraint import Constraint

class BLEU(Constraint):

    def __init__(self, max_bleu_score, compare_against_original=True):
        super().__init__(compare_against_original)
        if not isinstance(max_bleu_score, int):
            raise TypeError("max_bleu_score must be an int")
        self.max_bleu_score = max_bleu_score

    def _check_constraint(self, transformed_text, reference_text):
        ref = reference_text.words
        hyp = transformed_text.words
        bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
        return bleu_score <= self.max_bleu_score

