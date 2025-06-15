import math

from ..pre_transformation_constraint import PreTransformationConstraint


class MaxModificationRate(PreTransformationConstraint):

    def __init__(self, max_rate, min_threshold=1):
        assert isinstance(max_rate, float), "`max_rate` must be a float."
        assert max_rate >= 0 and max_rate <= 1, "`max_rate` must between 0 and 1."
        assert isinstance(min_threshold, int), "`min_threshold` must an int"

        self.max_rate = max_rate
        self.min_threshold = min_threshold

    def _get_modifiable_indices(self, current_text):

        threshold = max(
            math.ceil(current_text.num_words * self.max_rate), self.min_threshold
        )
        if len(current_text.attack_attrs["modified_indices"]) >= threshold:
            return set()
        else:
            return set(range(len(current_text.words)))
