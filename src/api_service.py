import re, os
from typing import List, Tuple, Union

from together import Together
from openai import OpenAI


class ServiceProvider:
    required_attributes = ["API_KEY"]

    def __init__(self, TaskData):
        self.client = self.Provider(api_key=self.API_KEY)
        self.TEMPLATE = TaskData.base_prompt

    def call_api(self, prompt: str, model: str, temperature: float = 0.6, n: int = 1, add_task_fewshot: bool = True) -> Tuple[str, int]:
        if add_task_fewshot:
            prompt = self.TEMPLATE + prompt
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.  \
                        Provide concise and accurate answers based on the given context and do not include irrelevant information.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            n=n,
            max_tokens=4096,
            temperature=temperature,
        )
        if n > 1:
            completion = [chat_completion.choices[i].message.content for i in range(n)]
        else:
            completion = chat_completion.choices[0].message.content
        total_tokens = chat_completion.usage.total_tokens
        return completion, total_tokens

    def calculate_cost(self, model: str, total_tokens: int) -> float:
        raise NotImplementedError("Subclasses must calculate costs in their own specific way.")


class TogetherAIAPI(ServiceProvider):
    Provider = Together
    API_KEY = os.getenv('TOGETHER_API_KEY')
    
    def calculate_cost(self, model: str, total_tokens: int) -> float:
        match = re.search(r'-(\d+)[Bb]-', model)
        model_size = int(match.group(1)) if match else 7  # Default to 7B if size not found

        # Check for "lite" or "turbo" in the model name
        is_lite = 'lite' in model.lower()
        is_turbo = 'turbo' in model.lower()
        is_moe = '8x22b' in model.lower() # hack for WizardLM MOE

        # Define the pricing tiers FOR LLaMa models
        price_tiers = {
            3: {'turbo': 0.06},
            8: {'lite': 0.10, 'turbo': 0.18, 'standard': 0.20},
            70: {'lite': 0.54, 'turbo': 0.88, 'standard': 0.90},
            405: {'turbo': 5.00}  # No 'lite' or 'standard' pricing for 405B
        }

        # Determine the price based on LlaMa model size and type
        if model_size in price_tiers:
            if is_lite:
                price_per_million = price_tiers[model_size].get('lite')
            elif is_turbo:
                price_per_million = price_tiers[model_size].get('turbo')
            else:
                price_per_million = price_tiers[model_size].get('standard')
        elif is_moe: # hack for wizardLM
            price_per_million = 1.20
        else:
            # Fallback to a default pricing structure if size is not explicitly listed
            price_per_million = next(
                price for size, price in [
                    (4, 0.10),
                    (8, 0.20),
                    (21, 0.30),
                    (41, 0.80),
                    (80, 0.90),
                    (110, 1.80),
                    (float('inf'), 2.40)
                ] if model_size <= size
            )
        
        return (total_tokens / 1_000_000) * price_per_million

class OpenAIAPI(ServiceProvider):
    Provider = OpenAI
    API_KEY = os.getenv("OPENAI_API_KEY")

    def calculate_cost(self, model: str, total_tokens: int) -> float:
        pass