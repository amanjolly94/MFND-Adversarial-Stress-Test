from .sentence_encoder import SentenceEncoder

import tensorflow_hub as hub

class UniversalSentenceEncoder(SentenceEncoder):

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/3"

        self._tfhub_url = tfhub_url
        # Lazily load the model
        self.model = None

    def encode(self, sentences):
        if not self.model:
            self.model = hub.load(self._tfhub_url)
        encoding = self.model(sentences)

        if isinstance(encoding, dict):
            encoding = encoding["outputs"]

        return encoding.numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None