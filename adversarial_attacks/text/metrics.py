from abc import ABC

import torch
import nltk

from .constraints.semantics.bert_score import BERTScore
from .constraints.semantics.sentence_encoders.bert import BERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Perplexity(ABC):
    def __init__(self, model_name="gpt2"):
        self.all_metrics = {}
        self.original_candidates = []
        self.successful_candidates = []

        if model_name == "gpt2":
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self.ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.ppl_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.ppl_model.eval()
            self.max_length = self.ppl_model.config.n_positions
        else:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self.ppl_model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ppl_model.eval()
            self.max_length = self.ppl_model.config.max_position_embeddings

        self.stride = 512

    def calculate(self, results):

        self.results = results
        self.original_candidates_ppl = []
        self.successful_candidates_ppl = []

        for i, result in enumerate(self.results):
 
            self.original_candidates.append(
                result.original_result.attacked_text.text.lower()
            )
            self.successful_candidates.append(
                result.perturbed_result.attacked_text.text.lower()
            )

        ppl_orig = self.calc_ppl(self.original_candidates)
        ppl_attack = self.calc_ppl(self.successful_candidates)

        self.all_metrics["avg_original_perplexity"] = round(ppl_orig, 2)

        self.all_metrics["avg_attack_perplexity"] = round(ppl_attack, 2)

        return self.all_metrics

    def calc_ppl(self, texts):
        with torch.no_grad():
            text = " ".join(texts)
            eval_loss = []
            input_ids = torch.tensor(
                self.ppl_tokenizer.encode(text, add_special_tokens=True)
            ).unsqueeze(0)
            # Strided perplexity calculation from huggingface.co/transformers/perplexity.html
            for i in range(0, input_ids.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, input_ids.size(1))
                trg_len = end_loc - i
                input_ids_t = input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids_t.clone()
                target_ids[:, :-trg_len] = -100

                outputs = self.ppl_model(input_ids_t, labels=target_ids)
                log_likelihood = outputs[0] * trg_len

                eval_loss.append(log_likelihood)

        return torch.exp(torch.stack(eval_loss).sum() / end_loc).item()
    
class USEMetric(ABC):
    def __init__(self, **kwargs):
        self.use_obj = UniversalSentenceEncoder()
        self.use_obj.model = UniversalSentenceEncoder()
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):

        self.results = results

        for i, result in enumerate(self.results):

            self.original_candidates.append(result.original_result.attacked_text)
            self.successful_candidates.append(result.perturbed_result.attacked_text)

        use_scores = []
        for c in range(len(self.original_candidates)):
            use_scores.append(
                self.use_obj._sim_score(
                    self.original_candidates[c], self.successful_candidates[c]
                ).item()
            )

        self.all_metrics["avg_attack_use_score"] = round(
            sum(use_scores) / len(use_scores), 2
        )

        return self.all_metrics
    
class BERTScoreMetric(ABC):
    def __init__(self, **kwargs):
        self.use_obj = BERTScore(
            min_bert_score=0.5, model_name="microsoft/deberta-large-mnli", num_layers=18
        )
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):

        self.results = results

        for i, result in enumerate(self.results):

            self.original_candidates.append(result.original_result.attacked_text)
            self.successful_candidates.append(result.perturbed_result.attacked_text)

        sbert_scores = []
        for c in range(len(self.original_candidates)):
            sbert_scores.append(
                self.use_obj._sim_score(
                    self.original_candidates[c], self.successful_candidates[c]
                )
            )

        self.all_metrics["avg_attack_bert_score"] = round(
            sum(sbert_scores) / len(sbert_scores), 2
        )

        return self.all_metrics
    
class SBERTMetric(ABC):
    def __init__(self, **kwargs):
        self.use_obj = BERT(model_name="all-MiniLM-L6-v2", metric="cosine")
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):

        self.results = results

        for i, result in enumerate(self.results):

            self.original_candidates.append(result.original_result.attacked_text)
            self.successful_candidates.append(result.perturbed_result.attacked_text)

        sbert_scores = []
        for c in range(len(self.original_candidates)):
            sbert_scores.append(
                self.use_obj._sim_score(
                    self.original_candidates[c], self.successful_candidates[c]
                ).item()
            )

        self.all_metrics["avg_attack_sentence_bert_similarity"] = round(
            sum(sbert_scores) / len(sbert_scores), 2
        )

        return self.all_metrics
    
class MeteorMetric(ABC):
    def __init__(self, **kwargs):
        self.original_candidates = []
        self.successful_candidates = []
        self.all_metrics = {}

    def calculate(self, results):

        self.results = results

        for i, result in enumerate(self.results):

            self.original_candidates.append(
                result.original_result.attacked_text.text
            )
            self.successful_candidates.append(
                result.perturbed_result.attacked_text.text
            )

        meteor_scores = []
        for c in range(len(self.original_candidates)):
            meteor_scores.append(
                nltk.translate.meteor(
                    [nltk.word_tokenize(self.original_candidates[c])],
                    nltk.word_tokenize(self.successful_candidates[c]),
                )
            )

        self.all_metrics["avg_attack_meteor_score"] = round(
            sum(meteor_scores) / len(meteor_scores), 2
        )

        return self.all_metrics