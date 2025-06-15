from ..pre_transformation_constraint import PreTransformationConstraint

class MaxWordIndexModification(PreTransformationConstraint):

    def __init__(self, max_length):
        self.max_length = max_length

    def _get_modifiable_indices(self, current_text):

        return set(range(min(self.max_length, len(current_text.words))))