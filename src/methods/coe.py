from time import time
from typing import List, Tuple, Union
from collections import Counter

from .base_cascade import CascadeMethod
from .utils import extract_answer


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
  