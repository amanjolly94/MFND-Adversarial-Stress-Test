from abc import ABC
import math

import numpy as np
import torch

from ...constraint import Constraint

class SentenceEncoder(Constraint, ABC):

    def __init__(
        self,
        threshold=0.8,
        metric="cosine",
        compare_against_original=True,
        window_size=None,
        skip_text_shorter_than_window=False,
    ):
        super().__init__(compare_against_original)
        self.metric = metric
        self.threshold = threshold
        self.window_size = window_size
        self.skip_text_shorter_than_window = skip_text_shorter_than_window

        if not self.window_size:
            self.window_size = float("inf")

        if metric == "cosine":
            self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        elif metric == "angular":
            self.sim_metric = get_angular_sim
        elif metric == "max_euclidean":
            # If the threshold requires embedding similarity measurement
            # be less than or equal to a certain value, just negate it,
            # so that we can still compare to the threshold using >=.
            self.threshold = -threshold
            self.sim_metric = get_neg_euclidean_dist
        else:
            raise ValueError(f"Unsupported metric {metric}.")

    def encode(self, sentences):

        raise NotImplementedError()

    def _sim_score(self, starting_text, transformed_text):

        try:
            modified_index = next(
                iter(transformed_text.attack_attrs["newly_modified_indices"])
            )
        except KeyError:
            raise KeyError(
                "Cannot apply sentence encoder constraint without `newly_modified_indices`"
            )
        starting_text_window = starting_text.text_window_around_index(
            modified_index, self.window_size
        )

        transformed_text_window = transformed_text.text_window_around_index(
            modified_index, self.window_size
        )

        starting_embedding, transformed_embedding = self.model.encode(
            [starting_text_window, transformed_text_window]
        )

        if not isinstance(starting_embedding, torch.Tensor):
            starting_embedding = torch.tensor(starting_embedding)

        if not isinstance(transformed_embedding, torch.Tensor):
            transformed_embedding = torch.tensor(transformed_embedding)

        starting_embedding = torch.unsqueeze(starting_embedding, dim=0)
        transformed_embedding = torch.unsqueeze(transformed_embedding, dim=0)

        return self.sim_metric(starting_embedding, transformed_embedding)

    def _score_list(self, starting_text, transformed_texts):

        # Return an empty tensor if transformed_texts is empty.
        # This prevents us from calling .repeat(x, 0), which throws an
        # error on machines with multiple GPUs (pytorch 1.2).
        if len(transformed_texts) == 0:
            return torch.tensor([])

        if self.window_size:
            starting_text_windows = []
            transformed_text_windows = []
            for transformed_text in transformed_texts:
                # @TODO make this work when multiple indices have been modified
                try:
                    modified_index = next(
                        iter(transformed_text.attack_attrs["newly_modified_indices"])
                    )
                except KeyError:
                    raise KeyError(
                        "Cannot apply sentence encoder constraint without `newly_modified_indices`"
                    )
                starting_text_windows.append(
                    starting_text.text_window_around_index(
                        modified_index, self.window_size
                    )
                )
                transformed_text_windows.append(
                    transformed_text.text_window_around_index(
                        modified_index, self.window_size
                    )
                )
            embeddings = self.encode(starting_text_windows + transformed_text_windows)
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)
            starting_embeddings = embeddings[: len(transformed_texts)]
            transformed_embeddings = embeddings[len(transformed_texts) :]
        else:
            starting_raw_text = starting_text.text
            transformed_raw_texts = [t.text for t in transformed_texts]
            embeddings = self.encode([starting_raw_text] + transformed_raw_texts)
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings)

            starting_embedding = embeddings[0]

            transformed_embeddings = embeddings[1:]

            # Repeat original embedding to size of perturbed embedding.
            starting_embeddings = starting_embedding.unsqueeze(dim=0).repeat(
                len(transformed_embeddings), 1
            )

        return self.sim_metric(starting_embeddings, transformed_embeddings)

    def _check_constraint_many(self, transformed_texts, reference_text):

        scores = self._score_list(reference_text, transformed_texts)

        for i, transformed_text in enumerate(transformed_texts):
            # Optionally ignore similarity score for sentences shorter than the
            # window size.
            if (
                self.skip_text_shorter_than_window
                and len(transformed_text.words) < self.window_size
            ):
                scores[i] = 1
            transformed_text.attack_attrs["similarity_score"] = scores[i].item()
        mask = (scores >= self.threshold).cpu().numpy().nonzero()
        return np.array(transformed_texts)[mask]

    def _check_constraint(self, transformed_text, reference_text):
        if (
            self.skip_text_shorter_than_window
            and len(transformed_text.words) < self.window_size
        ):
            score = 1
        else:
            score = self._sim_score(reference_text, transformed_text)

        transformed_text.attack_attrs["similarity_score"] = score
        return score >= self.threshold

def get_angular_sim(emb1, emb2):

    cos_sim = torch.nn.CosineSimilarity(dim=1)(emb1, emb2)
    return 1 - (torch.acos(cos_sim) / math.pi)


def get_neg_euclidean_dist(emb1, emb2):

    return -torch.sum((emb1 - emb2) ** 2, dim=1)