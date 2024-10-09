import re
from typing import List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from time import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .base_cascade import CascadeMethod
from .utils import extract_answer, calculate_accuracy


class AutoMix(CascadeMethod):
    def __init__(
            self,
            ServiceProvider,
            TaskData,
            cascade_tier_models: List[str],
            temperature: float = 0.6,
            num_bins: int = 8,
            routing_strategy: str = "threshold",
            train: bool = True
    ):
        super().__init__(ServiceProvider, TaskData, cascade_tier_models, temperature)
        self.num_bins = num_bins
        self.gap = 1 / num_bins
        self.routing_strategy = routing_strategy
        self.tools = {}
        if train:
            print("Training AutoMix's routing function...")
            start = time()
            self.train_router()
            self.setup_latency = time() - start
            print("Training complete!")
            print("Results: ", self.tools)
        else: # use self consistency instead
            pass # write load model logic later

    def train_router(self):
        for tier in range(self.n_tiers - 1):
            print(f"Training the routing function for {self.cascade_models[tier]} and {self.cascade_models[tier+1]}...")
            self._process_data_for_training(tier) # generate initial inference for training router
            self.tools[tier] = {}
            self.tools[tier]['FOLDER'] = f"automix_router_logs/{self.Task.data_url.split('/')[-1]}/"
            self.tools[tier]['best_param'] = self._train_router() # either with threshold or pomdp
        self.setup_cost = self.total_cost
        self.total_cost = 0


    def _process_data_for_training(self, tier, len_data=None):
        if not len_data: 
            len_data = min(75, self.Task.train_data.num_rows) ### CONFIRM LATER
            print("Training samples for router set to ", len_data)
        temp_data = self.Task.train_data.select(range(len_data)).train_test_split(test_size=0.33)
        self._temp_df, self._temp_val = temp_data['train'], temp_data['test']

        print("Generating inference on data subset for training...")
        self._temp_df = self._generate_label_process_data(tier, self._temp_df)
        # self._temp_val = self._generate_label_process_data(tier, self._temp_val) # Not needed for now
        print("Done with initial inference generation. Starting router training...!")


    def _generate_label_process_data(self, tier, data):
        prompts = data[self.Task.query_column]
        slm_responses = [self.generate_inference(prompt, self.cascade_models[tier]) for prompt in prompts]
        llm_responses = [self.generate_inference(prompt, self.cascade_models[tier+1]) for prompt in prompts]
        verifier_scores = self._self_verify(prompts, slm_responses, self.cascade_models[tier])
        
        labels = data[self.Task.label_column]
        if self.Task.groundtruth_need_regex:
            labels = extract_answer(labels, self.Task.label_regex)
        
        slm_perf = [int(str(l) == r) for l, r in zip(labels, extract_answer(slm_responses, self.Task.label_regex))]
        llm_perf = [int(str(l) == r) for l, r in zip(labels, extract_answer(llm_responses, self.Task.label_regex))]
        
        return pd.DataFrame({
            'prompt': prompts,
            'slm_response': slm_responses,
            'llm_response': llm_responses,
            'verifier_score': verifier_scores,
            'slm_perf': slm_perf,
            'llm_perf': llm_perf
        })

    def _train_router(self):
        if self.routing_strategy == "threshold":
            return self._train_threshold_router()
        elif self.routing_strategy == "pomdp":
            return self._train_pomdp_router()
        else:
            print("Invalid routing strategy; using self_consistency instead.")
            return 0.5

    def _train_threshold_router(self):
        thresholds = np.linspace(0, 1, self.num_bins + 1)
        best_threshold, best_performance = 0, float('-inf')
        for threshold in thresholds:
            performance = self._evaluate_threshold(threshold)
            if performance > best_performance:
                best_performance = performance
                best_threshold = threshold
        return best_threshold

    def _train_pomdp_router(self):
        self.tools['categories'] = ['NEEDY', 'GOOD', 'HOPELESS']
        self._categorize_rows()
        obs_probs = self._compute_obs_probs()
        action_seqs = []
        for reward in range(5000):
            action_array = np.array([[-reward, 0, -reward-1], [-100, -100, -reward-100]])
            actions = []
            for i in range(self.num_bins + 1):
                scores = obs_probs[i] * action_array
                actions.append(np.argmax(scores.sum(axis=1)))
            action_seqs.append(tuple(actions))
        return max(set(action_seqs), key=action_seqs.count)
    
    def _categorize_rows(self):
        p_10_slm = self._temp_df['slm_perf'].quantile(0.10)
        p_10_llm = self._temp_df['llm_perf'].quantile(0.10)
        conditions = [
            (self._temp_df['slm_perf'] <= self._temp_df['llm_perf']) & (self._temp_df['slm_perf'] != self._temp_df['llm_perf']),
            (self._temp_df['slm_perf'] == self._temp_df['llm_perf']) & (self._temp_df['slm_perf'] != 0),
            (self._temp_df['slm_perf'] <= p_10_slm) & (self._temp_df['llm_perf'] <= p_10_llm)
        ]
        self._temp_df['category'] = np.select(conditions, self.tools['categories'], default='UNDEFINED')

    def _compute_obs_probs(self):
        obs_probs = np.zeros((self.num_bins + 1, 3))
        for idx, prob in enumerate([i * self.gap for i in range(self.num_bins + 1)]):
            df_new = self._temp_df[(self._temp_df['verifier_score'] - prob).abs() < self.gap / 2]
            try:
                vcs = df_new['category'].value_counts()
                obs_probs[idx] = [(vcs[cat] if cat in vcs else 0) / len(df_new) for cat in self.tools['categories']]
            except Exception:
                pass
        return obs_probs
    
    def _evaluate_threshold(self, threshold: float):
        to_retry = self._temp_df['verifier_score'] <= threshold
        performances = np.where(to_retry, self._temp_df['llm_perf'], self._temp_df['slm_perf'])
        return performances.mean()

    def _self_verify(self, prompts: List[str], answers: List[str], model: str) -> float:
        if isinstance(prompts, str): # for single-value cases
            prompts, answers = [prompts], [answers]
        verifier_scores = []
        for prompt, answer in zip(prompts, answers):
            verifier_prompt = self._make_verifier_input(prompt, answer)
            # K times = 8
            verifier_responses = self.generate_inference(verifier_prompt, model, temp=1.0, n=8, add_task_fewshot=False)
            score = self._compute_verification_score(verifier_responses)
            print(verifier_responses)
            verifier_scores.append(score)
        return verifier_scores

    def _make_verifier_input(self, prompt: str, answer: str) -> str:
        with open("src/prompt_templates/automix_self_verifier.txt", "r") as f:
            verifier_fewshot = f.read()
        verifier_input = verifier_fewshot.format(prompt=prompt, answer=answer)
        verifier_input_as_list = verifier_input.split(" ")# to check for context length limit
        if len(verifier_input_as_list) > 8100: 
            verifier_input_as_list = verifier_input_as_list[-8100:]
            verifier_input = " ".join(verifier_input_as_list)
        
        return verifier_input

    def _compute_verification_score(self, verifier_responses: str) -> float:
        # Define a regex pattern to capture the verification decision
        decision_pattern = re.compile(r"Verification Decision:\s*\[Decision:\s*(Correct|Incorrect)\]", re.IGNORECASE)
        total_valid = 0
        correct_count = 0
        for item in verifier_responses:
            match = decision_pattern.search(item)
            if match:
                total_valid += 1
                if match.group(1).lower() == "correct":
                    correct_count += 1
        print("valid and correct,", total_valid, correct_count)
        if total_valid == 0:
            return 0
        return correct_count / total_valid

    def _inference_cascade(self, prompts: List[str]) -> Tuple[List[str], float]:
        answers = []
        for prompt in prompts:
            start_time = time()
            for tier in range(self.n_tiers):
                response = self.generate_inference(prompt=prompt, models=self.cascade_models[tier])
                if tier != self.n_tiers - 1: # not doing this for the last tier
                    ### consider moving `extract_answer` here
                    verifier_score = self._self_verify(prompt, response, self.cascade_models[tier])[0]
                    if self.routing_strategy == "threshold":
                        if verifier_score > self.tools[tier]["best_param"]:
                            break
                    elif self.routing_strategy == "pomdp":
                        # action = self.router.get_action(verifier_score, self.tools[tier]["best_param"])
                        print("Length of best param", len(self.tools[tier]["best_param"]))
                        action = self.tools[tier]["best_param"][self._get_nearest_prob_idx(verifier_score)]
                        if action == 0:
                            break
                    print(response)
            print("Exiting at tier ", tier)
            f_response = extract_answer(response, self.Task.label_regex)[0] ### consider moving before
            answers.append(f_response)
            self.total_latency += time() - start_time
        print(f"\nSetup cost in $$: {self.setup_cost}\nSetup latency: {self.setup_latency}")
        return answers, self.total_latency / len(prompts)
    
    def _get_nearest_prob_idx(self, prob: float) -> int:
        # return min(int(prob // self.gap), self.num_bins)
        x = min(int(prob // self.gap), self.num_bins)
        print ("Prob minus self.gap", prob // self.gap)
        print("Nearest prob idx", x)
        return x
    
    def evaluate(self, test_data):
        """Function to calculate all those extra AutoMix Performance stuff.
        Here we are using real costs instead of AutoMix's assumed costs.
        Need to fix later
        """
        predictions, avg_latency = self._inference_cascade(test_data[self.Task.query_column])
        accuracy = calculate_accuracy(predictions, test_data[self.Task.label_column])
        
        slm_performance = np.mean([self.Task.evaluate(self.generate_inference(prompt, self.cascade_models[0])) for prompt in test_data[self.Task.query_column]])
        llm_performance = np.mean([self.Task.evaluate(self.generate_inference(prompt, self.cascade_models[1])) for prompt in test_data[self.Task.query_column]])
        
        automix_performance = accuracy
        automix_cost = self.total_cost / len(test_data)
        
        slm_cost = self.Service.calculate_cost(self.cascade_models[0], len(test_data))
        llm_cost = self.Service.calculate_cost(self.cascade_models[1], len(test_data))
        
        slm_llm_slope = (llm_performance - slm_performance) / (llm_cost - slm_cost)
        automix_slm_slope = (automix_performance - slm_performance) / (automix_cost - slm_cost)
        ibc_lift = (automix_slm_slope - slm_llm_slope) / slm_llm_slope
        
        return {
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'total_cost': self.total_cost,
            'setup_cost': self.setup_cost,
            'ibc_lift': ibc_lift,
            'automix_slm_slope': automix_slm_slope,
            'automix_performance': automix_performance,
            'automix_cost': automix_cost,
        }