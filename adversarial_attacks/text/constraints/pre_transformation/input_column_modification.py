from ..pre_transformation_constraint import PreTransformationConstraint

class InputColumnModification(PreTransformationConstraint):

    def __init__(self, matching_column_labels, columns_to_ignore):
        self.matching_column_labels = matching_column_labels
        self.columns_to_ignore = columns_to_ignore

    def _get_modifiable_indices(self, current_text):

        if current_text.column_labels != self.matching_column_labels:
            return set(range(len(current_text.words)))

        idx = 0
        indices_to_modify = set()
        for column, words in zip(
            current_text.column_labels, current_text.words_per_input
        ):
            num_words = len(words)
            if column not in self.columns_to_ignore:
                indices_to_modify |= set(range(idx, idx + num_words))
            idx += num_words
        return indices_to_modify
