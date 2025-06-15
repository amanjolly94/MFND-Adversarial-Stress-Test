from abc import ABC, abstractmethod

class Constraint(ABC):

    def __init__(self, compare_against_original):
        self.compare_against_original = compare_against_original

    def call_many(self, transformed_texts, reference_text):

        incompatible_transformed_texts = []
        compatible_transformed_texts = []
        for transformed_text in transformed_texts:
            try:
                if self.check_compatibility(
                    transformed_text.attack_attrs["last_transformation"]
                ):
                    compatible_transformed_texts.append(transformed_text)
                else:
                    incompatible_transformed_texts.append(transformed_text)
            except KeyError:
                raise KeyError(
                    "transformed_text must have `last_transformation` attack_attr to apply constraint"
                )
        filtered_texts = self._check_constraint_many(
            compatible_transformed_texts, reference_text
        )
        return list(filtered_texts) + incompatible_transformed_texts

    def _check_constraint_many(self, transformed_texts, reference_text):

        return [
            transformed_text
            for transformed_text in transformed_texts
            if self._check_constraint(transformed_text, reference_text)
        ]

    def __call__(self, transformed_text, reference_text):

        try:
            if not self.check_compatibility(
                transformed_text.attack_attrs["last_transformation"]
            ):
                return True
        except KeyError:
            raise KeyError(
                "`transformed_text` must have `last_transformation` attack_attr to apply constraint."
            )
        return self._check_constraint(transformed_text, reference_text)

    @abstractmethod
    def _check_constraint(self, transformed_text, reference_text):

        raise NotImplementedError()

    def check_compatibility(self, transformation):

        return True

