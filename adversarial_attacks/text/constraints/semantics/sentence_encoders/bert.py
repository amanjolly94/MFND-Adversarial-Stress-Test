import torch
from sentence_transformers import SentenceTransformer

from .sentence_encoder import SentenceEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERT(SentenceEncoder):
    # https://huggingface.co/sentence-transformers
    def __init__(
        self,
        threshold=0.7,
        metric="cosine",
        model_name="bert-base-nli-stsb-mean-tokens",
        **kwargs
    ):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

    def encode(self, sentences):
        return self.model.encode(sentences)