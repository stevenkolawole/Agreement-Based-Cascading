from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from time import time
from collections import Counter
import re
from .frugalgpt_scorer import Scorer


class CascadeMethod:
    def __init__(
            self,
            ServiceProvider,
            TaskData,
            cascade_tier_models: List[Union[List[str], str]], # List of str or List of Lists denoting ensembles or single models cascades
            temperature: Union[List[float], float] = 0.6
    ):
        self.cascade_models = cascade_tier_models
        self.n_tiers = len(cascade_tier_models)
        self.temp = temperature
        self.Service = ServiceProvider
        self.Task = TaskData
        self.total_tokens = 0
        self.total_cost = 0
        self.total_latency = 0

    def generate_inference(self, prompt: str, models: Union[List[str], str]) -> Union[str, List[str]]:
        if isinstance(models, list): 
            # ensemble cascade
            results = self._process_in_parallel(models, lambda m: self.Service.call_api(prompt, m, self.temp))
            responses, tokens_list = zip(*results)
            cost_list = self._process_in_parallel(zip(models, tokens_list), 
                                                  lambda pair: self.Service.calculate_cost(pair[0], pair[1]))
            cost, tokens = sum(cost_list), sum(tokens_list)
        elif isinstance(self.temp, list): 
            # LLM Mixture-of-Thoughts Cascade
            results = self._process_in_parallel(self.temp, lambda t: self.Service.call_api(prompt, models, t))
            responses, tokens_list = zip(*results)
            tokens = sum(tokens_list)
            cost = self.Service.calculate_cost(models, tokens) # same model is used 3x
        else: 
            # Singular model and singular temperature
            responses, tokens = self.Service.call_api(prompt, models, self.temp)
            cost = self.Service.calculate_cost(models, tokens)

        self.total_tokens += tokens
        self.total_cost += cost
        return responses
    
    def _process_in_parallel(self, items, func) -> Union[List[str], Tuple[List[str], List[int]]]:
        with ThreadPoolExecutor(max_workers=len(list(items))) as executor:
            results = list(executor.map(func, items))
        return results
    
    def _inference_cascade(self):
        raise NotImplementedError("Subclasses must their cascade logic in their own specific way.")
    
    def inference_cascade(self, len_data: int = None):
        if len_data == None: len_data = min(100, self.Task.val_data.num_rows)
        
        prompts = self.Task.val_data[self.Task.query_column][:len_data]
        labels = self.Task.val_data[self.Task.label_column][:len_data]
        if self.Task.groundtruth_need_regex:
            labels = extract_answer(labels, self.Task.label_regex)

        print("Starting inference engine...")
        predictions, avg_latency = self._inference_cascade(prompts)
        print("Calculating accuracy with offline labels...")
        accuracy = calculate_accuracy(predictions, labels)
        return accuracy, avg_latency, self.total_cost
    

class EnsembleCascade(CascadeMethod):
    def __init__(
            self,
            ServiceProvider,
            TaskData,
            cascade_tier_models: List[Union[List[str], str]], 
            temperature: float = 0.6,
            agreement_threshold: float = 1.0 # (2/3) or full agreement = 1.0
    ):
        super().__init__(ServiceProvider, TaskData, cascade_tier_models, temperature)
        self._threshold = agreement_threshold

    def _inference_cascade(self, prompts: List[str]) -> Tuple[List[str], float]:
        """Loop through cascade tiers as needed; generate inference; measure latency as well.
        Dynamic enough to work for single models, single tiers situations as well"""
        answers = []
        for prompt in prompts:
            start_time = time()
            for tier in range(self.n_tiers):
                responses = self.generate_inference(prompt=prompt, models=self.cascade_models[tier])
                f_responses = extract_answer(responses, self.Task.label_regex)
                majority_answer, majority_count = Counter(f_responses).most_common(1)[0]
                consistency = (majority_count / len(f_responses)) >= self._threshold
                print(majority_answer, majority_count)
                print(responses)
                print(f_responses)
                if consistency and majority_answer != "": 
                    break
            print("Exiting at tier ", tier)
            self.total_latency += time() - start_time
            answers.append(majority_answer)
        return answers, self.total_latency / len(prompts)
    

class MOTLLMCascade(EnsembleCascade):
    def __init__(
        self,
        ServiceProvider,
        TaskData,
        cascade_tier_models: List[str],
        temperature: List[float] = [0.4, 0.6, 0.8],
        mixture_consistency_threshold: float = 2/3 # (2/3) or full consistency check = 1.0
    ):
        super().__init__(ServiceProvider, TaskData, cascade_tier_models, temperature)
        self._threshold = mixture_consistency_threshold


class FrugalGPT(CascadeMethod):
    def __init__(
            self,
            ServiceProvider,
            TaskData,
            cascade_tier_models: List[str], # List of str or List of Lists denoting ensembles or single models cascades
            temperature: float = 0.6,
            train: bool = False,
    ):
        super().__init__(ServiceProvider, TaskData, cascade_tier_models, temperature)
        self.tools = {}
        self._thresholds = (0.96, 0.37) # hardcoded from the paper
        if train:
            print("Training FrugalGPT's scorer functions...")
            start = time()
            self.train_scorer()
            self.setup_latency = time() - start
            print("Training complete!")
        else:
            pass # print("Loading complete!")

    def train_scorer(self):
        for tier in range(self.n_tiers):
            if tier != self.n_tiers - 1: # don't train for last tier
                print(f"Training the scoring function for {self.cascade_models[tier]}...")
                self.tools[tier] = {}
                self.tools[tier]['threshold'] = self._thresholds[tier]
                self.tools[tier]['FOLDER'] = f"scorer_logs/{self.Task.data_url.split('/')[-1]}/{tier}/"
                self.tools[tier]['Scorer'] = self._train_scorer(tier)
        self.setup_cost = self.total_cost # inference costs for training; does not include GPU cost
        self.total_cost = 0 # start total cost afresh

    def _train_scorer(self, tier):
        self._process_data_for_training(tier)
        return self._train_on_processed_data(tier)

    def _process_data_for_training(self, tier, len_data=None):
        if not len_data: 
            len_data = min(500, self.Task.train_data.num_rows)
            print("Training samples set to ", len_data)
        temp_data = self.Task.train_data.select(range(len_data)).train_test_split(test_size=.2)
        self._temp_train, self._temp_val = temp_data['train'], temp_data['test']
        print("Generating inference on data subset for training...")
        self._temp_train = self._generate_label_process_data(tier, self._temp_train)
        self._temp_val = self._generate_label_process_data(tier, self._temp_val)
    
    def _generate_label_process_data(self, tier, data, len_data=100):
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

    def _inference_cascade(self, prompts: List[str]) -> Tuple[List[str], float]:
        answers = []
        for prompt in prompts:
            start_time = time()
            for tier in range(self.n_tiers):
                response = self.generate_inference(prompt=prompt, models=self.cascade_models[tier])
                if tier != self.n_tiers - 1: # we don't need the check for the last tier
                    score = self.tools[tier]['Scorer'].get_score(response)
                    consistency = score > self.tools[tier]['threshold']
                    print(score)
                    print(response)
                    if consistency: 
                        break
            print("Exiting at tier ", tier)
            f_response = extract_answer(response, self.Task.label_regex)[0]
            answers.append(f_response)
            self.total_latency += time() - start_time
        print(f"\nSetup cost in $$: {self.setup_cost}\nSetup latency: {self.setup_latency}")
        return answers, self.total_latency / len(prompts)
    

class AutoMix(CascadeMethod):
    pass


def extract_answer(raw_responses: Union[List[str], str], regex_pattern: str) -> List[str]:
    responses = []
    if isinstance(raw_responses, str): # for cases where we have just a single model response
        raw_responses = [raw_responses]

    for r in raw_responses:
        try:
            matches = re.findall(regex_pattern, r)
            if matches:
                responses.append(matches[-1])
            else:
                print("Pattern match error!")
                responses.append("")
                
        except AttributeError:
            print(f"Answer not found in ==> `{r}`\n")
            responses.append("")
            
    return responses

def calculate_accuracy(pred: List[str], true_labels: List[str]) -> float:
    correct_count = sum(p == str(t) for p, t in zip(pred, true_labels))
    return correct_count / len(pred)
