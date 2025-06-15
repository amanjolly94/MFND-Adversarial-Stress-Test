import random
import string

import torch

from .base import BaseTransformation
from ..utils.word_embeddings import WordEmbedding

class WordSwap(BaseTransformation):

    def __init__(self, letters_to_insert=None):
        self.letters_to_insert = letters_to_insert
        if not self.letters_to_insert:
            self.letters_to_insert = string.ascii_letters

    def _get_replacement_words(self, word):

        raise NotImplementedError()

    def _get_random_letter(self):

        return random.choice(self.letters_to_insert)

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts
    
class WordSwapEmbedding(WordSwap):

    def __init__(self, max_candidates=15, embedding=None, **kwargs):
        super().__init__(**kwargs)
        if embedding is None:
            embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        self.max_candidates = max_candidates

        self.embedding = embedding

    def _get_replacement_words(self, word):

        try:
            word_id = self.embedding.word2index(word.lower())
            nnids = self.embedding.nearest_neighbours(word_id, self.max_candidates)
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = self.embedding.index2word(nbr_id)
                candidate_words.append(self.recover_word_case(nbr_word, word))
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []

    def recover_word_case(self, word, reference_word):
        """Makes the case of `word` like the case of `reference_word`.

        Supports lowercase, UPPERCASE, and Capitalized.
        """
        if reference_word.islower():
            return word.lower()
        elif reference_word.isupper() and len(reference_word) > 1:
            return word.upper()
        elif reference_word[0].isupper() and reference_word[1:].islower():
            return word.capitalize()
        else:
            # if other, just do not alter the word's case
            return word