from typing import List, Tuple, Union
from time import time
from typing import List, Tuple, Union
from collections import Counter

from .coe import EnsembleCascade
from .utils import extract_answer, normalize_answer


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

    def _inference_cascade(self, prompts: List[str]) -> Tuple[List[str], float]:
        """Loop through cascade tiers as needed; generate inference; measure latency as well.
        Dynamic enough to work for single models, single tiers situations as well"""
        answers = []
        for prompt in prompts:
            start_time = time()
            for tier in range(self.n_tiers):
                responses = self.generate_inference(prompt=prompt, models=self.cascade_models[tier])
                f_responses = extract_answer(responses, self.Task.label_regex)
                # if tier != (self.n_tiers - 1): # don't bother for the last layer
                majority_answer, majority_count = Counter(f_responses).most_common(1)[0]
                consistency = (majority_count / len(f_responses)) >= self._threshold
                if consistency and majority_answer != "": 
                    break
            # print("Exiting at tier ", tier)
            self.total_latency += time() - start_time
            answers.append(majority_answer)
        return answers, self.total_latency / len(prompts)