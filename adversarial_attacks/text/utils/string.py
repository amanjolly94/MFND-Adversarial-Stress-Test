import re
import string

import flair
import jieba
import stanza

def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    try:
        if re.search("[\u4e00-\u9FFF]", s):
            seg_list = jieba.cut(s, cut_all=False)
            s = " ".join(seg_list)
        else:
            s = " ".join(s.split())
    except Exception:
        s = " ".join(s.split())

    homos = """˗৭Ȣ𝟕бƼᏎƷᒿlO`ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"""
    exceptions = """'-_*@"""
    filter_pattern = homos + """'\\-_\\*@"""
    # TODO: consider whether one should add "." to `exceptions` (and "\." to `filter_pattern`)
    # example "My email address is xxx@yyy.com"
    filter_pattern = f"[\\w{filter_pattern}]+"
    words = []
    for word in s.split():
        # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the word.
        word = word.lstrip(exceptions)
        filt = [w.lstrip(exceptions) for w in re.findall(filter_pattern, word)]
        words.extend(filt)
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words

class FlairTokenizer(flair.data.Tokenizer):
    def tokenize(self, text: str):
        return words_from_text(text)
    
def flair_tag(sentence, tag_type="upos-fast"):
    """Tags a `Sentence` object using `flair` part-of-speech tagger."""
    global _flair_pos_tagger
    if not _flair_pos_tagger:
        from flair.models import SequenceTagger

        _flair_pos_tagger = SequenceTagger.load(tag_type)
    _flair_pos_tagger.predict(sentence, force_token_predictions=True)

def zip_flair_result(pred, tag_type="upos-fast"):
    """Takes a sentence tagging from `flair` and returns two lists, of words
    and their corresponding parts-of-speech."""
    from flair.data import Sentence

    if not isinstance(pred, Sentence):
        raise TypeError("Result from Flair POS tagger must be a `Sentence` object.")

    tokens = pred.tokens
    word_list = []
    pos_list = []
    for token in tokens:
        word_list.append(token.text)
        if "pos" in tag_type:
            pos_list.append(token.annotation_layers["upos"][0]._value)
        elif tag_type == "ner":
            pos_list.append(token.get_label("ner"))

    return word_list, pos_list

def zip_stanza_result(pred, tagset="universal"):

    if not isinstance(pred, stanza.models.common.doc.Document):
        raise TypeError("Result from Stanza POS tagger must be a `Document` object.")

    word_list = []
    pos_list = []

    for sentence in pred.sentences:
        for word in sentence.words:
            word_list.append(word.text)
            if tagset == "universal":
                pos_list.append(word.upos)
            else:
                pos_list.append(word.xpos)

    return word_list, pos_list