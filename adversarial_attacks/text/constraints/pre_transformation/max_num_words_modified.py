from ..pre_transformation_constraint import PreTransformationConstraint


class MaxNumWordsModified(PreTransformationConstraint):
    def __init__(self, max_num_words: int):
        self.max_num_words = max_num_words

    def _get_modifiable_indices(self, current_text):

        if len(current_text.attack_attrs["modified_indices"]) >= self.max_num_words:
            return set()
        else:
            return set(range(len(current_text.words)))
