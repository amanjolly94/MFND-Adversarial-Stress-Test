import torch

from adversarial_attacks.text.constraints.pre_transformation_constraint import PreTransformationConstraint
from adversarial_attacks.text.attacked_text import AttackedText, AttackResult

class AdversarialImageAttacks:

    def __init__(self, attack_name, model, device=None):

        self.attack_name = attack_name
        self.model = model
        if device:
            self.device = device
        else:
            self.device = next(model.parameters()).device

        self.targeted = False
        self.transforms = None

    def forward(self, inputs, labels=None, *args, **kwargs):
        raise NotImplementedError
    
    def get_target_label(self, inputs, labels=None):
        target_labels = labels
        return target_labels
    
    def get_logits(self, inputs, labels=None, *args, **kwargs):

        if self.transforms:
            inputs = self.transforms(inputs)

        logits = self.model(inputs)
        return logits
    
class AdversarialTextAttacks:

    def __init__(self, model, constraints, transformation, search_method):
        
        self.model = model
        self.search_method = search_method
        self.transformation = transformation

        self.constraints = []
        self.pre_transformation_constraints = []

        for constraint in constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)

        self.search_method.get_transformations = self.get_transformations
        # self.search_method.goal_function = self.goal_function
        self.search_method.get_goal_results = self.get_results
        self.search_method.get_indices_to_order = self.get_indices_to_order
        self.search_method.filter_transformations = self.filter_transformations
    
    def get_indices_to_order(self, current_text, **kwargs):
        indices_to_order = self.transformation(
            current_text,
            pre_transformation_constraints=self.pre_transformation_constraints,
            return_indices=True,
            **kwargs,
        )

        len_text = len(indices_to_order)

        # Convert indices_to_order to list for easier shuffling later
        return len_text, list(indices_to_order)
    
    def get_transformations(self, current_text, original_text=None, **kwargs):

        transformed_texts = self.transformation(
            current_text,
            pre_transformation_constraints=self.pre_transformation_constraints,
            **kwargs,
        )

        return self.filter_transformations(
            transformed_texts, current_text, original_text
        )
    
    def filter_transformations(self, transformed_texts, current_text, original_text=None):
        transformed_texts = [
            t for t in transformed_texts if t.text != current_text.text
        ]

        # transformed_texts = []
        filtered_texts = []

        # for transformed_text in transformed_texts:
        #     transformed_texts.append(transformed_text)

        filtered_texts = transformed_texts[:]
        for C in self.constraints:
            if C.compare_against_original:
                filtered_texts = C.call_many(filtered_texts, original_text)
            else:
                filtered_texts = C.call_many(filtered_texts, current_text)

        filtered_texts.sort(key=lambda t: t.text)
        return filtered_texts
    
    def _is_goal_complete(self, model_output, ground_truth_output, target_max_score=None):

        # target_max_score (float): If set, goal is to reduce model output to
        # below this score. Otherwise, goal is to change the overall predicted
        # class.
        if target_max_score:
            return model_output[ground_truth_output] < target_max_score
        elif (model_output.numel() == 1) and isinstance(ground_truth_output, float):
            return abs(ground_truth_output - model_output.item()) >= 0.5
        else:
            return model_output.argmax() != ground_truth_output


    
    def attack(self, example, ground_truth_output, target_max_score=None):
        example = AttackedText(example)

        self.num_queries = 0
        raw_output, batch_prob = self._call_model([example])
        goal_score = 1 - batch_prob[0][ground_truth_output]

        if self._is_goal_complete(raw_output, ground_truth_output, target_max_score):
            goal_status = 0
        else:
            # Searching
            goal_status = 1

        initial_result = AttackResult(
            example,
            raw_output,
            int(raw_output.argmax()),
            goal_status,
            goal_score,
            ground_truth_output,
        )

        # final_result = self.search_method(initial_result)

        return initial_result
    
    def _call_model(self, attacked_text_list):
        inputs = [at.tokenizer_input for at in attacked_text_list]
        raw_output = self.model(inputs)
        batch_prob = torch.nn.functional.softmax(raw_output, dim=1)
        
        return raw_output, batch_prob
    
    def get_results(self, attacked_text_list, check_skip=False):
        
        results = []
        self.num_queries += len(attacked_text_list)
        model_outputs, _ = self._call_model(attacked_text_list)

        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            # if self._is_goal_complete(raw_output, ground_truth_output, target_max_score):
            #     goal_status = 0
            # else:
            #     # Searching
            #     goal_status = 1

            print(attacked_text, raw_output)