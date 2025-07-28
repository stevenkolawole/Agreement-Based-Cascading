import pandas as pd
from src.dataloaders import OverrulingDataset, HeadlineDataset, CoQADataset, GSM8KDataset
from src.methods import AutoMix, EnsembleCascade, FrugalGPT, MOTLLMCascade
from src.api_service import TogetherAIAPI, OpenAIAPI
import os


os.environ['TOGETHER_API_KEY'] = 'your_together_api_key_here'   # Replace with your actual API key

ensemble_cascade_2level = [
    [
        'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
        # 'meta-llama/Llama-3.2-3B-Instruct-Turbo',
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'google/gemma-2-9b-it',
    ],
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
]
ensemble_cascade_3level = [
    [
        'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
        # 'meta-llama/Llama-3.2-3B-Instruct-Turbo',
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'google/gemma-2-9b-it',
    ],
    [
        'Qwen/Qwen2-72B-Instruct',
        'google/gemma-2-27b-it',
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
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


def run_single_model(api, task, model):
    print(f"Running inference on {model}...")
    single_run = EnsembleCascade(api, task, [model])
    accuracy, avg_latency, total_cost = single_run.inference_cascade() # add `len_data=10` as argument for testing
    return {"model": model.split('/')[-1], "accuracy": accuracy, "cost": total_cost, "avg_latency": avg_latency}

def run_cascade_method(method_class, api, task, models, **kwargs):
    method = method_class(api, task, models, **kwargs)
    accuracy, avg_latency, total_cost = method.inference_cascade() # add `len_data=10` as argument for testing
    return accuracy, avg_latency, total_cost

def evaluate_models(task_class, task_name, configs):
    task = task_class()
    api = TogetherAIAPI(TaskData=task)
    results = []

    # Evaluate single models
    for model in single_models:
        results.append(run_single_model(api, task, model))

    # Evaluate cascade methods
    for method_name, method_class, models, kwargs in configs:
        accuracy, avg_latency, total_cost = run_cascade_method(method_class, api, task, models, **kwargs)
        results.append({
            "model": method_name,
            "accuracy": accuracy,
            "cost": total_cost,
            "avg_latency": avg_latency,
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"cascade_results_{task_name}.csv", index=False)
    print(df_results)

def main():
    configs = [
        ("MoT-LLM Cascade 2-level", MOTLLMCascade, single_models_cascade_2level, {}),
        ("CoE 2-level", EnsembleCascade, ensemble_cascade_2level, {}),
        ("AutoMix_T 2-level", AutoMix, single_models_cascade_2level, {"routing_strategy": "threshold", "train": True}),
        ("AutoMix_P 2-level", AutoMix, single_models_cascade_2level, {"routing_strategy": "pomdp", "train": True}),
        ("FrugalGPT 2-level", FrugalGPT, single_models_cascade_2level, {"train": True}),
        ("MoT-LLM Cascade 3-level", MOTLLMCascade, single_models_cascade_3level, {}),
        ("CoE 3-level", EnsembleCascade, ensemble_cascade_3level, {}),
        ("AutoMix_T 3-level", AutoMix, single_models_cascade_3level, {"routing_strategy": "threshold", "train": True}),
        ("AutoMix_P 3-level", AutoMix, single_models_cascade_3level, {"routing_strategy": "pomdp", "train": True}),
        ("FrugalGPT 3-level", FrugalGPT, single_models_cascade_3level, {"train": True}),
    ]

    tasks = [
        (GSM8KDataset, "gsm8k"),
        (OverrulingDataset, "overruling"),
        (HeadlineDataset, "headlines"),
        (CoQADataset, "coqa"),
    ]

    for task_class, task_name in tasks:
        print(f"{task_name.upper()} Task now...")
        evaluate_models(task_class, task_name, single_models, configs)


if __name__ == "__main__":
    main()