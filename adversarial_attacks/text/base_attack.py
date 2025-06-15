
from abc import ABC, abstractmethod

from ..adv_attack import AdversarialTextAttacks


class BaseAttack(AdversarialTextAttacks, ABC):

    @staticmethod
    @abstractmethod
    def build(model_wrapper, **kwargs):

        raise NotImplementedError()
    

