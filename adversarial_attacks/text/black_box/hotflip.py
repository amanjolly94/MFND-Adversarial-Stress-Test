from ...adv_attack import AdversarialTextAttacks

from ..base_attack import BaseAttack

# Word swap
from ..transformations.word_swap import WordSwapEmbedding


class HotFlip(BaseAttack):
    # https://arxiv.org/abs/1712.06751
    @staticmethod
    def build(model_wrapper):

        transformation = WordSwapEmbedding(max_candidates=50)

        constraints = [
            RepeatModification(), # Don't modify the same word twice
            StopwordModification(), # Don't modify the stopwords
            MaxWordsPerturbed(max_num_words=2),
            WordEmbeddingDistance(min_cos_sim=0.8), # The cosine similarity between the embedding of words is bigger than 0.8
            PartOfSpeech(), # The two words have the same part-of-speech.
        ]

        search_method = BeamSearch(beam_width=10)

        return AdversarialTextAttacks(constraints, transformation, search_method)