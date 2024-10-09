from rich import print as rprint
import pandas as pd

from src.dataloaders import OverrulingDataset, HeadlineDataset, CoQADataset, GSM8KDataset
from src.methods import AutoMix, EnsembleCascade, FrugalGPT, MOTLLMCascade

from src.api_service import TogetherAIAPI, OpenAIAPI


import os
os.environ['TOGETHER_API_KEY'] = '5de421f4d56d44ac7400e98c3cac5dc98e184bc92e297e552aadd7198def0661'

ensemble_cascade_2level = [
    [
        'meta-llama/Llama-3.2-3B-Instruct-Turbo',
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        # 'google/gemma-2-9b-it',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ],
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
]

ensemble_cascade_3level = [
    [
        'meta-llama/Llama-3.2-3B-Instruct-Turbo',
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        # 'google/gemma-2-9b-it',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ],
    [
        'Qwen/Qwen2-72B-Instruct',
        'microsoft/WizardLM-2-8x22B',
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        # 'deepseek-ai/deepseek-llm-67b-chat', # can swap for `lite` if we want to reduce cost further
    ],
    'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
]
single_models = [x for i in ensemble_cascade_3level for x in ([i] if isinstance(i, str) else i)]

single_models_cascade_2level = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
]

single_models_cascade_3level = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
]


print("GSM8K Now...")

# Task1 = GSM8KDataset()
# API1 = TogetherAIAPI(TaskData=Task1)
# results = []
# for model in single_models:
#     print(f"Running inference on {model}...")
#     single_run = EnsembleCascade( 
#         # ensemble cascade works well for just a single model, if only one model is passed in
#         API1, Task1, [model],
#     )
#     accurracy, avg_latency, total_cost = single_run.inference_cascade()
#     print(accurracy, avg_latency, total_cost)
#     results.append({
#         "model": model.split('/')[-1],
#         "accuracy": accurracy,
#         "cost": total_cost,
#         "avg_latency": avg_latency,
#     })

# c_results = {}

# c_results['MoT-LLM Cascade 2-level'] = MOTLLMCascade(
#     ServiceProvider=API1,
#     TaskData=Task1,
#     cascade_tier_models=single_models_cascade_2level,
# ).inference_cascade()

# c_results['CoE 2-level'] = EnsembleCascade(
#     ServiceProvider=API1,
#     TaskData=Task1,
#     cascade_tier_models=ensemble_cascade_2level
# ).inference_cascade()

# c_results['AutoMix_T 2-level'] = AutoMix(API1, Task1, 
#     single_models_cascade_2level,
#     routing_strategy="threshold", # or "pomdp",
#     train=True
# ).inference_cascade()

# c_results['AutoMix_P 2-level'] = AutoMix(API1, Task1, 
#     single_models_cascade_2level,
#     routing_strategy="pomdp", # or "pomdp",
#     train=True
# ).inference_cascade()

# c_results['FrugalGPT 2-level'] = FrugalGPT(
#     API1, Task1, single_models_cascade_2level, train=True
# ).inference_cascade()

# c_results['MoT-LLM Cascade 3-level'] = MOTLLMCascade(
#     ServiceProvider=API1,
#     TaskData=Task1,
#     cascade_tier_models=single_models_cascade_3level,
# ).inference_cascade()

# c_results['CoE 3-level'] = EnsembleCascade(
#     ServiceProvider=API1,
#     TaskData=Task1,
#     cascade_tier_models=ensemble_cascade_3level
# ).inference_cascade()

# c_results['AutoMix_T 3-level'] = AutoMix(API1, Task1, 
#     single_models_cascade_3level,
#     routing_strategy="threshold", # or "pomdp",
#     train=True
# ).inference_cascade()

# c_results['AutoMix_P 3-level'] = AutoMix(API1, Task1, 
#     single_models_cascade_3level,
#     routing_strategy="pomdp", # or "pomdp",
#     train=True
# ).inference_cascade()

# c_results['FrugalGPT 3-level'] = FrugalGPT(
#     API1, Task1, single_models_cascade_3level, train=True
# ).inference_cascade()

# for k, v in c_results.items():
#     results.append({
#     "model": k,
#     "accuracy": v[0],
#     "cost": v[2],
#     "avg_latency": v[1],
# })

# df_results = pd.DataFrame(results) 

# df_results.to_csv("single_models_gsm8k.csv", index=False)
# print(df_results)


print("Overruling TAsk now...")

Task2 = OverrulingDataset()
API2 = TogetherAIAPI(TaskData=Task2)

results = []

for model in single_models:
    print(f"Running inference on {model}...")
    single_run = EnsembleCascade( 
        # ensemble cascade works well for just a single model, if only one model is passed in
        API2, Task2, [model],
    )
    accurracy, avg_latency, total_cost = single_run.inference_cascade()
    print(accurracy, avg_latency, total_cost)
    results.append({
        "model": model.split('/')[-1],
        "accuracy": accurracy,
        "cost": total_cost,
        "avg_latency": avg_latency,
    })

c_results = {}

c_results['MoT-LLM Cascade 2-level'] = MOTLLMCascade(
    ServiceProvider=API2,
    TaskData=Task2,
    cascade_tier_models=single_models_cascade_2level,
).inference_cascade()

c_results['CoE 2-level'] = EnsembleCascade(
    ServiceProvider=API2,
    TaskData=Task2,
    cascade_tier_models=ensemble_cascade_2level
).inference_cascade()

c_results['AutoMix_T 2-level'] = AutoMix(API2, Task2, 
    single_models_cascade_2level,
    routing_strategy="threshold", # or "pomdp",
    train=True
).inference_cascade()

c_results['AutoMix_P 2-level'] = AutoMix(API2, Task2, 
    single_models_cascade_2level,
    routing_strategy="pomdp", # or "pomdp",
    train=True
).inference_cascade()

c_results['FrugalGPT 2-level'] = FrugalGPT(
    API2, Task2, single_models_cascade_2level, train=True
).inference_cascade()

c_results['MoT-LLM Cascade 3-level'] = MOTLLMCascade(
    ServiceProvider=API2,
    TaskData=Task2,
    cascade_tier_models=single_models_cascade_3level,
).inference_cascade()

c_results['CoE 3-level'] = EnsembleCascade(
    ServiceProvider=API2,
    TaskData=Task2,
    cascade_tier_models=ensemble_cascade_3level
).inference_cascade()

c_results['AutoMix_T 3-level'] = AutoMix(API2, Task2, 
    single_models_cascade_3level,
    routing_strategy="threshold", # or "pomdp",
    train=True
).inference_cascade()

c_results['AutoMix_P 3-level'] = AutoMix(API2, Task2, 
    single_models_cascade_3level,
    routing_strategy="pomdp", # or "pomdp",
    train=True
).inference_cascade()

c_results['FrugalGPT 3-level'] = FrugalGPT(
    API2, Task2, single_models_cascade_3level, train=True
).inference_cascade()

for k, v in c_results.items():
    results.append({
    "model": k,
    "accuracy": v[0],
    "cost": v[2],
    "avg_latency": v[1],
})

df_results = pd.DataFrame(results) 

df_results.to_csv("single_models_overruling.csv", index=False)
print(df_results)


print("HEADLINE TASK now...")
Task3 = HeadlineDataset()
API3 = TogetherAIAPI(TaskData=Task3)

results = []

for model in single_models:
    print(f"Running inference on {model}...")
    single_run = EnsembleCascade( 
        # ensemble cascade works well for just a single model, if only one model is passed in
        API3, Task3, [model],
    )
    accurracy, avg_latency, total_cost = single_run.inference_cascade()
    print(accurracy, avg_latency, total_cost)
    results.append({
        "model": model.split('/')[-1],
        "accuracy": accurracy,
        "cost": total_cost,
        "avg_latency": avg_latency,
    })

c_results = {}

c_results['MoT-LLM Cascade 2-level'] = MOTLLMCascade(
    ServiceProvider=API3,
    TaskData=Task3,
    cascade_tier_models=single_models_cascade_2level,
).inference_cascade()

c_results['CoE 2-level'] = EnsembleCascade(
    ServiceProvider=API3,
    TaskData=Task3,
    cascade_tier_models=ensemble_cascade_2level
).inference_cascade()

c_results['AutoMix_T 2-level'] = AutoMix(API3, Task3, 
    single_models_cascade_2level,
    routing_strategy="threshold", # or "pomdp",
    train=True
).inference_cascade()

c_results['AutoMix_P 2-level'] = AutoMix(API3, Task3, 
    single_models_cascade_2level,
    routing_strategy="pomdp", # or "pomdp",
    train=True
).inference_cascade()

c_results['FrugalGPT 2-level'] = FrugalGPT(
    API3, Task3, single_models_cascade_2level, train=True
).inference_cascade()

c_results['MoT-LLM Cascade 3-level'] = MOTLLMCascade(
    ServiceProvider=API3,
    TaskData=Task3,
    cascade_tier_models=single_models_cascade_3level,
).inference_cascade()

c_results['CoE 3-level'] = EnsembleCascade(
    ServiceProvider=API3,
    TaskData=Task3,
    cascade_tier_models=ensemble_cascade_3level
).inference_cascade()

c_results['AutoMix_T 3-level'] = AutoMix(API3, Task3, 
    single_models_cascade_3level,
    routing_strategy="threshold", # or "pomdp",
    train=True
).inference_cascade()

c_results['AutoMix_P 3-level'] = AutoMix(API3, Task3, 
    single_models_cascade_3level,
    routing_strategy="pomdp", # or "pomdp",
    train=True
).inference_cascade()

c_results['FrugalGPT 3-level'] = FrugalGPT(
    API3, Task3, single_models_cascade_3level, train=True
).inference_cascade()

for k, v in c_results.items():
    results.append({
    "model": k,
    "accuracy": v[0],
    "cost": v[2],
    "avg_latency": v[1],
})

df_results = pd.DataFrame(results)

df_results.to_csv("single_models_headlines.csv", index=False)
print(df_results)


print("COQA Task now...")

Task4 = CoQADataset()
API4 = TogetherAIAPI(TaskData=Task4)

results = []

for model in single_models:
    print(f"Running inference on {model}...")
    single_run = EnsembleCascade( 
        # ensemble cascade works well for just a single model, if only one model is passed in
        API4, Task4, [model],
    )
    accurracy, avg_latency, total_cost = single_run.inference_cascade()
    print(accurracy, avg_latency, total_cost)
    results.append({
        "model": model.split('/')[-1],
        "accuracy": accurracy,
        "cost": total_cost,
        "avg_latency": avg_latency,
    })

c_results = {}

c_results['MoT-LLM Cascade 2-level'] = MOTLLMCascade(
    ServiceProvider=API4,
    TaskData=Task4,
    cascade_tier_models=single_models_cascade_2level,
).inference_cascade()

c_results['CoE 2-level'] = EnsembleCascade(
    ServiceProvider=API4,
    TaskData=Task4,
    cascade_tier_models=ensemble_cascade_2level
).inference_cascade()

c_results['AutoMix_T 2-level'] = AutoMix(API4, Task4, 
    single_models_cascade_2level,
    routing_strategy="threshold", # or "pomdp",
    train=True
).inference_cascade()

c_results['AutoMix_P 2-level'] = AutoMix(API4, Task4, 
    single_models_cascade_2level,
    routing_strategy="pomdp", # or "pomdp",
    train=True
).inference_cascade()

c_results['FrugalGPT 2-level'] = FrugalGPT(
    API4, Task4, single_models_cascade_2level, train=True
).inference_cascade()

c_results['MoT-LLM Cascade 3-level'] = MOTLLMCascade(
    ServiceProvider=API4,
    TaskData=Task4,
    cascade_tier_models=single_models_cascade_3level,
).inference_cascade()

c_results['CoE 3-level'] = EnsembleCascade(
    ServiceProvider=API4,
    TaskData=Task4,
    cascade_tier_models=ensemble_cascade_3level
).inference_cascade()

c_results['AutoMix_T 3-level'] = AutoMix(API4, Task4, 
    single_models_cascade_3level,
    routing_strategy="threshold", # or "pomdp",
    train=True
).inference_cascade()

c_results['AutoMix_P 3-level'] = AutoMix(API4, Task4, 
    single_models_cascade_3level,
    routing_strategy="pomdp", # or "pomdp",
    train=True
).inference_cascade()

c_results['FrugalGPT 3-level'] = FrugalGPT(
    API4, Task4, single_models_cascade_3level, train=True
).inference_cascade()

for k, v in c_results.items():
    results.append({
    "model": k,
    "accuracy": v[0],
    "cost": v[2],
    "avg_latency": v[1],
})

df_results = pd.DataFrame(results)

df_results.to_csv("single_models_coqa.csv", index=False)
print(df_results)
