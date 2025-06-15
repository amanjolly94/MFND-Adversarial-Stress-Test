from ..pre_transformation_constraint import PreTransformationConstraint

class RepeatModification(PreTransformationConstraint):

    def _get_modifiable_indices(self, current_text):

        try:
            return (
                set(range(len(current_text.words)))
                - current_text.attack_attrs["modified_indices"]
            )
        except KeyError:
            raise KeyError(
                "`modified_indices` in attack_attrs required for RepeatModification constraint."
            )
