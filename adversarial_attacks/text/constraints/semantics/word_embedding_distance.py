from ..constraint import Constraint
from ...utils.word_embeddings import WordEmbedding

class WordEmbeddingDistance(Constraint):
    def __init__(
        self,
        embedding=None,
        include_unknown_words=True,
        min_cos_sim=None,
        max_mse_dist=None,
        cased=False,
        compare_against_original=True,
    ):
        super().__init__(compare_against_original)
        if embedding is None:
            embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        self.include_unknown_words = include_unknown_words
        self.cased = cased

        if bool(min_cos_sim) == bool(max_mse_dist):
            raise ValueError("You must choose either `min_cos_sim` or `max_mse_dist`.")
        self.min_cos_sim = min_cos_sim
        self.max_mse_dist = max_mse_dist

        self.embedding = embedding

    def get_cos_sim(self, a, b):
        return self.embedding.get_cos_sim(a, b)

    def get_mse_dist(self, a, b):
        return self.embedding.get_mse_dist(a, b)

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        # FIXME The index i is sometimes larger than the number of tokens - 1
        if any(
            i >= len(reference_text.words) or i >= len(transformed_text.words)
            for i in indices
        ):
            return False

        for i in indices:
            ref_word = reference_text.words[i]
            transformed_word = transformed_text.words[i]

            if not self.cased:
                # If embedding vocabulary is all lowercase, lowercase words.
                ref_word = ref_word.lower()
                transformed_word = transformed_word.lower()

            try:
                ref_id = self.embedding.word2index(ref_word)
                transformed_id = self.embedding.word2index(transformed_word)
            except KeyError:
                # This error is thrown if x or x_adv has no corresponding ID.
                if self.include_unknown_words:
                    continue
                return False

            # Check cosine distance.
            if self.min_cos_sim:
                cos_sim = self.get_cos_sim(ref_id, transformed_id)
                if cos_sim < self.min_cos_sim:
                    return False
            # Check MSE distance.
            if self.max_mse_dist:
                mse_dist = self.get_mse_dist(ref_id, transformed_id)
                if mse_dist > self.max_mse_dist:
                    return False

        return True