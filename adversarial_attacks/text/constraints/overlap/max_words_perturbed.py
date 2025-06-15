import math

from ..constraint import Constraint


class MaxWordsPerturbed(Constraint):


    def __init__(
        self, max_num_words=None, max_percent=None, compare_against_original=True
    ):
        super().__init__(compare_against_original)
        if not compare_against_original:
            raise ValueError(
                "Cannot apply constraint MaxWordsPerturbed with `compare_against_original=False`"
            )

        if (max_num_words is None) and (max_percent is None):
            raise ValueError("must set either `max_percent` or `max_num_words`")
        if max_percent and not (0 <= max_percent <= 1):
            raise ValueError("max perc must be between 0 and 1")
        self.max_num_words = max_num_words
        self.max_percent = max_percent

    def _check_constraint(self, transformed_text, reference_text):
        num_words_diff = len(transformed_text.all_words_diff(reference_text))
        if self.max_percent:
            min_num_words = min(len(transformed_text.words), len(reference_text.words))
            max_words_perturbed = math.ceil(min_num_words * (self.max_percent))
            max_percent_met = num_words_diff <= max_words_perturbed
        else:
            max_percent_met = True
        if self.max_num_words:
            max_num_words_met = num_words_diff <= self.max_num_words
        else:
            max_num_words_met = True

        return max_percent_met and max_num_words_met
