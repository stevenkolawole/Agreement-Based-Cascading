from typing import List, Tuple, Union
import re
from sklearn.metrics import f1_score


def extract_answer(raw_responses: Union[List[str], str], regex_pattern: str) -> List[str]:
    responses = []
    if isinstance(raw_responses, str): # for cases where we have just a single model response
        raw_responses = [raw_responses]

    for r in raw_responses:
        matches = re.findall(regex_pattern, r)
        if matches:
            responses.append(matches[-1])
        else:
            print("Pattern match error!")
            responses.append("")            
    return responses

def calculate_accuracy(preds: List[str], true_labels: List[str]) -> float:
    correct_count = sum(p.strip().lower() == str(t).strip().lower() 
                        for p, t in zip(preds, true_labels))
    return correct_count / len(preds)


def calculate_f1(preds: List[str], true_labels: List[str]) -> float:
    normalized_preds = [normalize_answer(pred) for pred in preds]
    normalized_labels = [normalize_answer(label) for label in true_labels]
    return f1_score(normalized_labels, normalized_preds, average="macro")

def normalize_answer(answer: str) -> str:
    answer = answer.lower()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = re.sub(r'\b(the|a|an)\b', '', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    return answer
