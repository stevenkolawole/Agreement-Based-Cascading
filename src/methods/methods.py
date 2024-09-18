from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from time import time
from collections import Counter
import re


class CascadeMethod:
    def __init__(
            self,
            ServiceProvider,
            TaskData,
            cascade_tier_models: List[Union[List[str], str]], # List of str or List of Lists denoting ensembles or single models cascades
            temperature: Union[List[str], str] = 0.6
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
        if len_data == None: len_data = self.Task.val_data.num_rows
        
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
            temperature: Union[List[str], str] = 0.6,
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
#                print(majority_answer, majority_count)
#                print(responses)
#                print(f_responses)
                if consistency and majority_answer != "": 
                    break
#            print("Exiting at tier ", tier)
            self.total_latency += time() - start_time
            answers.append(majority_answer)
        return answers, self.total_latency / len(prompts)
    

class MOTLLMCascade(EnsembleCascade):
    def __init__(
        self,
        ServiceProvider,
        TaskData,
        cascade_tier_models: List[Union[List[str], str]],
        temperature: Union[List[str], str] = [0.4, 0.6, 0.8],
        mixture_consistency_threshold: float = 2/3 # (2/3) or full consistency check = 1.0
    ):
        super().__init__(ServiceProvider, TaskData, cascade_tier_models, temperature)
        self._threshold = mixture_consistency_threshold


class FrugalGPT(CascadeMethod):
    pass


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
            print("Answer not found in ==> `{r}`\n\n") # add f-string later
            responses.append("")
            
    return responses

def calculate_accuracy(pred: List[str], true_labels: List[str]) -> float:
    correct_count = sum(p == str(t) for p, t in zip(pred, true_labels))
    return correct_count / len(pred)
