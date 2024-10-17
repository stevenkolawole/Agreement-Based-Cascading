from time import time
from typing import List, Tuple, Union
from collections import Counter
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer

ps = PorterStemmer()

from .base_cascade import CascadeMethod
from .utils import extract_answer, normalize_answer
from ..dataloaders import CoQADataset # need it to check for F1


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
                if isinstance(self.Task, CoQADataset): # more nuanced measure of consistency is needed
                    consistency_score = calculate_f1_for_text_similarity(f_responses)
                    # print("Consistency score:", consistency_score, f_responses)
                    consistency = consistency_score >= 1.0 # hardcoded for now
                    majority_answer = max(set(f_responses), key=f_responses.count)
                else:
                    majority_answer, majority_count = Counter(f_responses).most_common(1)[0]
                    consistency = (majority_count / len(f_responses)) >= self._threshold
                if consistency and majority_answer != "": 
                    break
            # print("Exiting at tier ", tier)
            self.total_latency += time() - start_time
            answers.append(majority_answer)
        return answers, self.total_latency / len(prompts)


def calculate_f1_for_text_similarity(responses: List[str]) -> float:
    def calculate_overlap(s1: set, s2: set) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    normalized_responses = [normalize_answer(resp) for resp in responses]
    tokenized_responses = [set(ps.stem(word) for word in resp.split()) for resp in normalized_responses]
    
    n = len(tokenized_responses)
    if n <= 1:
        return 1.0  # Perfect consistency for 0 or 1 response
    
    total_similarity = 0
    comparisons = 0
    
    for i in range(n):
        for j in range(i+1, n):
            similarity = calculate_overlap(tokenized_responses[i], tokenized_responses[j])
            total_similarity += similarity
            comparisons += 1
    
    average_similarity = total_similarity / comparisons if comparisons > 0 else 0
    return average_similarity
    
def calculate_f1_for_coqa(responses: List[str], average="macro") -> float:
    normalized_responses = [normalize_answer(resp) for resp in responses]
    n = len(normalized_responses)
    
    # Create all pairwise comparisons
    labels, references = [], []
    for i in range(n):
        for j in range(i+1, n):  # Start from i+1 to avoid self-comparison and duplicates
            labels.append(normalized_responses[i])
            references.append(normalized_responses[j])
            # Add the reverse comparison as well
            labels.append(normalized_responses[j])
            references.append(normalized_responses[i])
    
    # Convert to binary vectors for each unique answer
    unique_answers = list(set(normalized_responses))
    binary_labels = [[1 if l == ua else 0 for ua in unique_answers] for l in labels]
    binary_references = [[1 if r == ua else 0 for ua in unique_answers] for r in references]
    
    # Calculate F1 score
    f1 = f1_score(binary_references, binary_labels, average=average, zero_division=1)
    
    return f1
