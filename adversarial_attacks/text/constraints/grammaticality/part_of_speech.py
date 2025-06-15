import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import nltk
import stanza
import torch

from ..constraint import Constraint
from ...utils.string import FlairTokenizer, zip_flair_result, zip_stanza_result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

flair.device = device

class PartOfSpeech(Constraint):

    def __init__(
        self,
        tagger_type="nltk",
        tagset="universal",
        allow_verb_noun_swap=True,
        compare_against_original=True,
        language_nltk="eng",
        language_stanza="en",
    ):
        super().__init__(compare_against_original)
        self.tagger_type = tagger_type
        self.tagset = tagset
        self.allow_verb_noun_swap = allow_verb_noun_swap
        self.language_nltk = language_nltk
        self.language_stanza = language_stanza

        if tagger_type == "flair":
            if tagset == "universal":
                self._flair_pos_tagger = SequenceTagger.load("upos-fast")
            else:
                self._flair_pos_tagger = SequenceTagger.load("pos-fast")

        if tagger_type == "stanza":
            self._stanza_pos_tagger = stanza.Pipeline(
                lang=self.language_stanza,
                processors="tokenize, pos",
                tokenize_pretokenized=True,
            )

    def _can_replace_pos(self, pos_a, pos_b):
        return (pos_a == pos_b) or (
            self.allow_verb_noun_swap and set([pos_a, pos_b]) <= set(["NOUN", "VERB"])
        )

    def _get_pos(self, before_ctx, word, after_ctx):
        context_words = before_ctx + [word] + after_ctx
        context_key = " ".join(context_words)

        if self.tagger_type == "nltk":
            word_list, pos_list = zip(
                *nltk.pos_tag(
                    context_words, tagset=self.tagset, lang=self.language_nltk
                )
            )

        if self.tagger_type == "flair":
            context_key_sentence = Sentence(
                context_key,
                use_tokenizer=FlairTokenizer(),
            )
            self._flair_pos_tagger.predict(context_key_sentence)
            word_list, pos_list = zip_flair_result(
                context_key_sentence
            )

        if self.tagger_type == "stanza":
            word_list, pos_list = zip_stanza_result(
                self._stanza_pos_tagger(context_key), tagset=self.tagset
            )


        # idx of `word` in `context_words`
        assert word in word_list, "POS list not matched with original word list."
        word_idx = word_list.index(word)
        return pos_list[word_idx]

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        for i in indices:
            reference_word = reference_text.words[i]
            transformed_word = transformed_text.words[i]
            before_ctx = reference_text.words[max(i - 4, 0) : i]
            after_ctx = reference_text.words[
                i + 1 : min(i + 4, len(reference_text.words))
            ]
            ref_pos = self._get_pos(before_ctx, reference_word, after_ctx)
            replace_pos = self._get_pos(before_ctx, transformed_word, after_ctx)
            if not self._can_replace_pos(ref_pos, replace_pos):
                return False

        return True
