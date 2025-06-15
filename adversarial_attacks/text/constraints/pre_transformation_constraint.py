from abc import ABC, abstractmethod

class PreTransformationConstraint(ABC):

    def __call__(self, current_text, transformation):

        if not self.check_compatibility(transformation):
            return set(range(len(current_text.words)))
        return self._get_modifiable_indices(current_text)

    @abstractmethod
    def _get_modifiable_indices(current_text):
 
        raise NotImplementedError()

    def check_compatibility(self, transformation):

        return True