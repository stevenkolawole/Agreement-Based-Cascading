from typing import List, Tuple, Union
import re


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
