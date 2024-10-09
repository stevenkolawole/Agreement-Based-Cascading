from typing import List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from time import time

from .utils import extract_answer, calculate_accuracy, calculate_f1
from ..dataloaders import CoQADataset # need it to check for F1

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

    def generate_inference(self, prompt: str, 
                           models: Union[List[str], str],
                           temp: Union[List[float], float] = None,
                           n: str = 1,
                           add_task_fewshot=True,
                           ) -> Union[str, List[str]]:
        if temp == None: temp = self.temp
        if isinstance(models, list): 
            # ensemble cascade
            results = self._process_in_parallel(models, lambda m: self.Service.call_api(prompt, m, temp))
            responses, tokens_list = zip(*results)
            cost_list = self._process_in_parallel(zip(models, tokens_list), 
                                                  lambda pair: self.Service.calculate_cost(pair[0], pair[1]))
            cost, tokens = sum(cost_list), sum(tokens_list)
        elif isinstance(temp, list): 
            # LLM Mixture-of-Thoughts Cascade
            results = self._process_in_parallel(temp, lambda t: self.Service.call_api(prompt, models, t))
            responses, tokens_list = zip(*results)
            tokens = sum(tokens_list)
            cost = self.Service.calculate_cost(models, tokens) # same model is used 3x
        else: 
            # Singular model and singular temperature
            responses, tokens = self.Service.call_api(prompt, models, temp, n=n, add_task_fewshot=add_task_fewshot)
            cost = self.Service.calculate_cost(models, tokens)

        self.total_tokens += tokens
        self.total_cost += cost
        return responses
    
    def _process_in_parallel(self, items, func) -> Union[List[str], Tuple[List[str], List[int]]]:
        with ThreadPoolExecutor(max_workers=len(list(items))) as executor:
            results = list(executor.map(func, items))
        return results

    def inference_cascade(self, len_data: int = None):
        if len_data == None: len_data = min(100, self.Task.val_data.num_rows)
        
        prompts = self.Task.val_data[self.Task.query_column][:len_data]
        labels = self.Task.val_data[self.Task.label_column][:len_data]
        if self.Task.groundtruth_need_regex:
            labels = extract_answer(labels, self.Task.label_regex)

        print("Starting inference engine...")
        predictions, avg_latency = self._inference_cascade(prompts)
        if isinstance(self.Task, CoQADataset):
            print("Calculating F1-score with offline labels...")
            accuracy = calculate_f1(predictions, labels)
        else:
            print("Calculating accuracy with offline labels...")
            accuracy = calculate_accuracy(predictions, labels)
        return accuracy, avg_latency, self.total_cost
        
    def _inference_cascade(self):
        raise NotImplementedError("Subclasses must their cascade logic in their own specific way.")
    