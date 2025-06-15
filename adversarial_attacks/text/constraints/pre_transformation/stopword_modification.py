import nltk

from ..pre_transformation_constraint import PreTransformationConstraint

class StopwordModification(PreTransformationConstraint):

    def __init__(self, stopwords=None, language="english"):
        if stopwords is not None:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(nltk.corpus.stopwords.words(language))

    def _get_modifiable_indices(self, current_text):

        non_stopword_indices = set()
        for i, word in enumerate(current_text.words):
            if word not in self.stopwords:
                non_stopword_indices.add(i)
        return non_stopword_indices

    def check_compatibility(self, transformation):

        return True
