import nltk

from ..constraint import Constraint


class METEOR(Constraint):

    def __init__(self, max_meteor, compare_against_original=True):
        super().__init__(compare_against_original)
        if not isinstance(max_meteor, int):
            raise TypeError("max_meteor must be an int")
        self.max_meteor = max_meteor

    def _check_constraint(self, transformed_text, reference_text):
        meteor = nltk.translate.meteor([reference_text], transformed_text)
        return meteor <= self.max_meteor
