from time import time
from typing import List, Tuple, Union
import numpy as np
from scipy import optimize

from .base_cascade import CascadeMethod
from .frugalgpt_scorer import Scorer
from .utils import extract_answer


class FrugalGPT(CascadeMethod):
    def __init__(
            self,
            ServiceProvider,
            TaskData,
            cascade_tier_models: List[str],
            temperature: float = 0.6,
            train: bool = True,
    ):
        super().__init__(ServiceProvider, TaskData, cascade_tier_models, temperature)
        self.tools = {}
        if train:
            print("Training FrugalGPT's scorer functions...")
            start = time()
            self._temp_scores, self._temp_accuracies = [], [] # to optimize threshold later
            self.train_scorer()
            self.setup_latency = time() - start
            print("Training complete!")
        else:
            pass # print("Loading complete!") # implement loading logic later

    def train_scorer(self):
        for tier in range(self.n_tiers):
            if tier != self.n_tiers - 1: # don't train for last tier
                print(f"Training the scoring function for {self.cascade_models[tier]}...")
                self.tools[tier] = {}
                self.tools[tier]['FOLDER'] = f"frugalgpt_scorer_logs/{self.Task.data_url.split('/')[-1]}/{tier}/"
                self.tools[tier]['Scorer'] = self._train_scorer(tier)
                self.tools[tier]['threshold'] = None # will be computed during optimization

        # After training all scorers, use the temp scores and accs to optimize the thresholds
        self._optimize_thresholds()

        self.setup_cost = self.total_cost # inference costs for training; does not include GPU cost
        self.total_cost = 0 # start total cost afresh

    def _train_scorer(self, tier):
        self._process_data_for_training(tier)
        scorer = self._train_on_processed_data(tier)
        self._construct_threshold_data(scorer) # Prepare data for optimization
        return scorer
    
    def _process_data_for_training(self, tier, len_data=None):
        if not len_data: 
            len_data = min(500, self.Task.train_data.num_rows)
            print("Training samples for router set to ", len_data)
        temp_data = self.Task.train_data.select(range(len_data)).train_test_split(test_size=.2)
        self._temp_train, self._temp_val = temp_data['train'], temp_data['test']
        print("Generating inference on data subset for training...")
        self._temp_train = self._generate_label_process_data(tier, self._temp_train)
        self._temp_val = self._generate_label_process_data(tier, self._temp_val)
    
    def _generate_label_process_data(self, tier, data):
        raw_responses = []
        prompts = []
        for prompt in data[self.Task.query_column]:
            prompts.append(prompt)
            response = self.generate_inference(prompt, self.cascade_models[tier])
            raw_responses.append(response)
        preds = extract_answer(raw_responses, self.Task.label_regex)

        labels = data[self.Task.label_column]
        if self.Task.groundtruth_need_regex:
            labels = extract_answer(labels, self.Task.label_regex)

        def compute_quality(example, idx):
            return {
                "query": f"{example[self.Task.query_column]}\n{raw_responses[idx]}",
                "label": int(str(labels[idx]) == preds[idx])
            }
        data = data.map(
            compute_quality, 
            with_indices=True,
            remove_columns=data.column_names
        )
        return data

    def _train_on_processed_data(self, tier):
        scorer = Scorer(TASK_FOLDER=self.tools[tier]['FOLDER'])
        scorer.pipeline(self._temp_train,
                        self._temp_val,
                        "query",)
        return scorer
    
    def _construct_threshold_data(self, scorer):
        tier_scores = [scorer.get_score(r) for r in self._temp_val["query"]]
        self._temp_scores.append(tier_scores)        
        self._temp_accuracies.append(self._temp_val["label"])
    
    def _optimize_thresholds(self):
        # Perform optimization
        thresholds = self._optimize(self._temp_scores, self._temp_accuracies)
        print("THRESHOLDS;", thresholds)
        # Set the optimized thresholds
        for tier in range(self.n_tiers - 1):
            self.tools[tier]['threshold'] = thresholds[tier]

    def _optimize(self, scores, accuracies):
        def objective(thresholds):
            total_accuracy = 0
            n_samples = len(scores[0])
            for i in range(n_samples):
                for tier in range(len(thresholds)):
                    if scores[tier][i] >= thresholds[tier]:
                        total_accuracy += accuracies[tier][i]
                        break
                else:
                    # If no threshold is met, use the last tier
                    total_accuracy += accuracies[-1][i] # should it be '-1' or [tier+1]?
            return -total_accuracy / n_samples  # Negative because we want to maximize

        # Initial guess: use median scores as starting thresholds
        initial_thresholds = [np.median(tier_scores) for tier_scores in scores]
        # Optimize
        result = optimize.minimize(objective, initial_thresholds, method='Nelder-Mead')
        return result.x

    def _inference_cascade(self, prompts: List[str]) -> Tuple[List[str], float]:
        answers = []
        for prompt in prompts:
            start_time = time()
            for tier in range(self.n_tiers):
                response = self.generate_inference(prompt=prompt, models=self.cascade_models[tier])
                if tier != self.n_tiers - 1: # we don't need the check for the last tier
                    score = self.tools[tier]['Scorer'].get_score(f"{prompt}\n{response}")
                    consistency = score > self.tools[tier]['threshold']
                    if consistency: 
                        break
            # print("Exiting at tier ", tier)
            f_response = extract_answer(response, self.Task.label_regex)[0]
            answers.append(f_response)
            self.total_latency += time() - start_time
        print(f"\nSetup cost in $$: {self.setup_cost}\nSetup latency: {self.setup_latency}")
        return answers, self.total_latency / len(prompts)
