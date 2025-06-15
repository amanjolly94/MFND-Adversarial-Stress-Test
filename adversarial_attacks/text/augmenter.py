import random
import tqdm

from adversarial_attacks.text.constraints.pre_transformation_constraint import PreTransformationConstraint
from .attacked_text import AttackedText
from .metrics import Perplexity, SBERTMetric

class Augmenter:

    def __init__(
        self,
        transformation,
        constraints=[],
        pct_words_to_swap=0.1,
        transformations_per_example=1,
        high_yield=False,
        fast_augment=False,
        enable_advanced_metrics=False,
    ):

        self.transformation = transformation
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example

        self.constraints = []
        self.pre_transformation_constraints = []
        self.high_yield = high_yield
        self.fast_augment = fast_augment
        self.advanced_metrics = enable_advanced_metrics
        for constraint in constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)

    def _filter_transformations(self, transformed_texts, current_text, original_text):

        for C in self.constraints:
            if len(transformed_texts) == 0:
                break
            if C.compare_against_original:

                transformed_texts = C.call_many(transformed_texts, original_text)
            else:
                transformed_texts = C.call_many(transformed_texts, current_text)
        return transformed_texts

    def augment(self, text):

        attacked_text = AttackedText(text)
        original_text = attacked_text
        all_transformed_texts = set()
        num_words_to_swap = max(
            int(self.pct_words_to_swap * len(attacked_text.words)), 1
        )
        augmentation_results = []
        for _ in range(self.transformations_per_example):
            current_text = attacked_text
            words_swapped = len(current_text.attack_attrs["modified_indices"])

            while words_swapped < num_words_to_swap:
                transformed_texts = self.transformation(
                    current_text, self.pre_transformation_constraints
                )

                # Get rid of transformations we already have
                transformed_texts = [
                    t for t in transformed_texts if t not in all_transformed_texts
                ]

                # Filter out transformations that don't match the constraints.
                transformed_texts = self._filter_transformations(
                    transformed_texts, current_text, original_text
                )

                # if there's no more transformed texts after filter, terminate
                if not len(transformed_texts):
                    break

                # look for all transformed_texts that has enough words swapped
                if self.high_yield or self.fast_augment:
                    ready_texts = [
                        text
                        for text in transformed_texts
                        if len(text.attack_attrs["modified_indices"])
                        >= num_words_to_swap
                    ]
                    for text in ready_texts:
                        all_transformed_texts.add(text)
                    unfinished_texts = [
                        text for text in transformed_texts if text not in ready_texts
                    ]

                    if len(unfinished_texts):
                        current_text = random.choice(unfinished_texts)
                    else:
                        # no need for further augmentations if all of transformed_texts meet `num_words_to_swap`
                        break
                else:
                    current_text = random.choice(transformed_texts)

                # update words_swapped based on modified indices
                words_swapped = max(
                    len(current_text.attack_attrs["modified_indices"]),
                    words_swapped + 1,
                )

            all_transformed_texts.add(current_text)

            # when with fast_augment, terminate early if there're enough successful augmentations
            if (
                self.fast_augment
                and len(all_transformed_texts) >= self.transformations_per_example
            ):
                if not self.high_yield:
                    all_transformed_texts = random.sample(
                        all_transformed_texts, self.transformations_per_example
                    )
                break

        perturbed_texts = sorted([at.printable_text() for at in all_transformed_texts])

        if self.advanced_metrics:
            for transformed_texts in all_transformed_texts:
                augmentation_results.append(
                    AugmentationResult(original_text, transformed_texts)
                )
            perplexity_stats = Perplexity().calculate(augmentation_results)
            sbert_stats = SBERTMetric().calculate(augmentation_results)
            return perturbed_texts, perplexity_stats, sbert_stats

        return perturbed_texts

    def augment_many(self, text_list, show_progress=False):
        """Returns all possible augmentations of a list of strings according to
        ``self.transformation``.

        Args:
            text_list (list(string)): a list of strings for data augmentation
        Returns a list(string) of augmented texts.
        :param show_progress: show process during augmentation
        """
        if show_progress:
            text_list = tqdm.tqdm(text_list, desc="Augmenting data...")
        return [self.augment(text) for text in text_list]

    def augment_text_with_ids(self, text_list, id_list, show_progress=True):
        """Supplements a list of text with more text data.

        Returns the augmented text along with the corresponding IDs for
        each augmented example.
        """
        if len(text_list) != len(id_list):
            raise ValueError("List of text must be same length as list of IDs")
        if self.transformations_per_example == 0:
            return text_list, id_list
        all_text_list = []
        all_id_list = []
        if show_progress:
            text_list = tqdm.tqdm(text_list, desc="Augmenting data...")
        for text, _id in zip(text_list, id_list):
            all_text_list.append(text)
            all_id_list.append(_id)
            augmented_texts = self.augment(text)
            all_text_list.extend
            all_text_list.extend([text] + augmented_texts)
            all_id_list.extend([_id] * (1 + len(augmented_texts)))
        return all_text_list, all_id_list



class AugmentationResult:
    def __init__(self, text1, text2):
        self.original_result = self.tempResult(text1)
        self.perturbed_result = self.tempResult(text2)

    class tempResult:
        def __init__(self, text):
            self.attacked_text = text