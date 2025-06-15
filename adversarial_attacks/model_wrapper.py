import torch
import numpy as np

class ImageModelWrapper:
    def __init__(self, model, transforms):
        self.model = model
        self.transforms = transforms

    def transform(self, pil_img):

        return self.transforms(pil_img).unsqueeze(0)
    
    def __call__(self, img):
        pass

class PyTorchModelWrapper:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input_list, batch_size=32):
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer(text_input_list)
        ids = torch.tensor(ids).to(model_device)

        with torch.no_grad():
            outputs = self.batch_model_predict(
                self.model, ids, batch_size=batch_size
            )

        return outputs
    
    def batch_model_predict(self, model_predict, inputs, batch_size=32):

        outputs = []
        i = 0
        while i < len(inputs):
            batch = inputs[i : i + batch_size]
            batch_preds = model_predict(batch)

            batch_preds = batch_preds.cpu()

            # Cast all predictions iterables to ``np.ndarray`` types.
            if not isinstance(batch_preds, np.ndarray):
                batch_preds = np.array(batch_preds)
            outputs.append(batch_preds)
            i += batch_size

        return np.concatenate(outputs, axis=0)

    def get_grad(self, text_input, loss_fn=torch.nn.CrossEntropyLoss()):

        if not hasattr(self.model, "get_input_embeddings"):
            raise AttributeError(
                f"{type(self.model)} must have method `get_input_embeddings` that returns `torch.nn.Embedding` object that represents input embedding layer"
            )
        if not isinstance(loss_fn, torch.nn.Module):
            raise ValueError("Loss function must be of type `torch.nn.Module`.")

        self.model.train()

        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        ids = self.tokenizer([text_input])
        ids = torch.tensor(ids).to(model_device)

        predictions = self.model(ids)

        output = predictions.argmax(dim=1)
        loss = loss_fn(predictions, output)
        loss.backward()

        # grad w.r.t to word embeddings

        # Fix for Issue #601

        # Check if gradient has shape [max_sequence,1,_] ( when model input in transpose of input sequence)

        if emb_grads[0].shape[1] == 1:
            grad = torch.transpose(emb_grads[0], 0, 1)[0].cpu().numpy()
        else:
            # gradient has shape [1,max_sequence,_]
            grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": ids[0].tolist(), "gradient": grad}

        return output

    def _tokenize(self, inputs):
 
        return [self.tokenizer.convert_ids_to_tokens(self.tokenizer(x)) for x in inputs]
    
    def tokenize(self, inputs, strip_prefix=False):

        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            # TODO: Find a better way to identify prefixes. These depend on the model, so cannot be resolved in ModelWrapper.

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        return tokens
    

class HuggingFaceModelWrapper:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def __call__(self, text_input_list):

        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        return outputs.logits
    
    def get_grad(self, text_input):

        self.model.train()
        embedding_layer = self.model.get_input_embeddings()
        original_state = embedding_layer.weight.requires_grad
        embedding_layer.weight.requires_grad = True

        emb_grads = []

        def grad_hook(module, grad_in, grad_out):
            emb_grads.append(grad_out[0])

        emb_hook = embedding_layer.register_backward_hook(grad_hook)

        self.model.zero_grad()
        model_device = next(self.model.parameters()).device
        input_dict = self.tokenizer(
            [text_input],
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        input_dict.to(model_device)
        predictions = self.model(**input_dict).logits

        try:
            labels = predictions.argmax(dim=1)
            loss = self.model(**input_dict, labels=labels)[0]
        except TypeError:
            raise TypeError(
                f"{type(self.model)} class does not take in `labels` to calculate loss. "
                "One cause for this might be if you instantiatedyour model using `transformer.AutoModel` "
                "(instead of `transformers.AutoModelForSequenceClassification`)."
            )

        loss.backward()

        # grad w.r.t to word embeddings
        grad = emb_grads[0][0].cpu().numpy()

        embedding_layer.weight.requires_grad = original_state
        emb_hook.remove()
        self.model.eval()

        output = {"ids": input_dict["input_ids"], "gradient": grad}

        return output
    
    def _tokenize(self, inputs):

        return [
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer([x], truncation=True)["input_ids"][0]
            )
            for x in inputs
        ]
    
    def tokenize(self, inputs, strip_prefix=False):

        tokens = self._tokenize(inputs)
        if strip_prefix:
            # `aux_chars` are known auxiliary characters that are added to tokens
            strip_chars = ["##", "Ġ", "__"]
            # TODO: Find a better way to identify prefixes. These depend on the model, so cannot be resolved in ModelWrapper.

            def strip(s, chars):
                for c in chars:
                    s = s.replace(c, "")
                return s

            tokens = [[strip(t, strip_chars) for t in x] for x in tokens]

        return tokens