# Agreement-Based Cascading (ABC) for Efficient Inference

[![TMLR](https://img.shields.io/badge/TMLR-July%202025-blue)](https://openreview.net/forum?id=jn9B7LMlzk)

This repository contains the implementation for **API-based experiments** from the ABC paper, comparing Agreement-Based Cascading with state-of-the-art methods like FrugalGPT, AutoMix, and MoT-LLM Cascade on black-box LLM APIs.

## 🎯 Overview

This codebase implements the experiments from **Section 5.2.3** of the ABC paper, demonstrating 2-25× cost reductions in API-based inference while maintaining competitive accuracy. ABC uses ensemble agreement (voting) as a training-free deferral mechanism for black-box model APIs.

## 🛠 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Together.ai API key ## can be updated directly in the `run` python scripts or notebook
export TOGETHER_API_KEY="your_api_key_here"
```

## 📊 Supported Datasets

- **GSM8K**: Math reasoning problems  
- **CoQA**: Conversational question answering  
- **OVERRULING**: Legal reasoning tasks  
- **HEADLINES**: Financial news sentiment classification

## 🚀 Quick Start

### Basic Usage

```bash
# Run all experiments (this will take time and cost money!)
python run.py

# Quick test with limited data
# Modify run.py line 49: add `len_data=10` parameter to inference_cascade()
# use run.ipynb for iterative testing/analysis.
```

### Manual Configuration

```python
from src.dataloaders import GSM8KDataset
from src.methods import EnsembleCascade  # This is ABC!
from src.api_service import TogetherAIAPI

# Setup
task = GSM8KDataset()
api = TogetherAIAPI(TaskData=task)

# Define cascade tiers
cascade_tiers = [
    # Tier 1: Ensemble of small models
    [
        'meta-llama/Meta-Llama-3-8B-Instruct-Lite',
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'google/gemma-2-9b-it'
    ],
    # Tier 2: Large model
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
]

# Run ABC
abc = EnsembleCascade(
    api, 
    task, 
    cascade_tiers,
    agreement_threshold=1.0 
)

accuracy, avg_latency, total_cost = abc.inference_cascade()
```

## 🔧 Key Components

### ABC Implementation (`src/methods/coe.py`)

- **`EnsembleCascade`**: Main ABC implementation using voting-based agreement  
- **Agreement Threshold**: Configurable consensus requirement (1.0 = unanimous, 0.67 = 2/3 majority)  
- **Smart Deferral**: Uses different agreement metrics for different tasks (F1-based for CoQA, voting for others)

### API Service (`src/api_service.py`)

- **`TogetherAIAPI`**: Together.ai integration with automatic cost calculation  
- **Error Handling**: Robust retry mechanism for API failures  
- **Cost Tracking**: Automatic per-token cost calculation based on model pricing

### Baseline Methods (`src/methods/`)

- **`AutoMix`**: Threshold and POMDP-based routing with self-verification  
- **`FrugalGPT`**: DistilBERT-based cascade routing  
- **`MOTLLMCascade`**: Mixture-of-Thoughts consistency checking

## 📈 Experiment Configuration

The main experiment in `run.py` compares:

1. **Single Models**: All individual models for baseline comparison  
2. **ABC (CoE)**: 2-level and 3-level ensemble cascades  
3. **Baseline Methods**: 2-level and 3-level of AutoMix, FrugalGPT, MoT-LLM Cascade systems

### Model Tiers Used

```python
# 2-Level Cascade
Tier 1: ['Llama-3-8B-Lite', 'Llama-3.1-8B-Turbo', 'Gemma-2-9B']
Tier 2: 'Llama-3.1-70B-Turbo'

# 3-Level Cascade  
Tier 1: ['Llama-3-8B-Lite', 'Llama-3.1-8B-Turbo', 'Gemma-2-9B']
Tier 2: ['Qwen2-72B', 'Gemma-2-27B', 'Llama-3.1-70B-Turbo']
Tier 3: 'Llama-3.1-405B-Turbo'
```

## 💰 Cost Analysis

The codebase automatically tracks:

- **Total API Cost**: Based on Together.ai pricing (per million tokens)  
- **Token Usage**: Input + output tokens for each API call  
- **Latency**: End-to-end inference time per sample

Example cost breakdown (per million tokens):

- Llama-3-8B-Lite: $0.10  
- Llama-3.1-8B-Turbo: $0.18  
- Llama-3.1-70B-Turbo: $0.88  
- Llama-3.1-405B-Turbo: $5.00

## 📊 Results

Results are saved as CSV files. Each contains accuracy, cost, and latency metrics for all methods.

## ⚙️ Customization

### Adjusting Agreement Threshold

```python
# Stricter agreement (fewer early exits, higher accuracy)
abc = EnsembleCascade(api, task, cascade_tiers, agreement_threshold=1.0)

# Relaxed agreement (more early exits, lower cost)
abc = EnsembleCascade(api, task, cascade_tiers, agreement_threshold=0.67)
```

---

### Testing Mode

For development/testing, limit the dataset size:

```python
# In run.py, line 49, add len_data parameter:
accuracy, avg_latency, total_cost = method.inference_cascade(len_data=10)
```

---

### Adding New Datasets

Create a new dataset class in `src/dataloaders.py`:

```python
class YourDataset(Dataset):
    data_url = "huggingface/dataset-name"
    query_column = "input_text"
    label_column = "label"
    PROMPT_PREFIX_FILE = "src/prompt_templates/your_task.txt"
```

Add the prompt template in `src/prompt_templates/your_task.txt`

---

### Adding New API Providers

The current implementation supports Together.ai with a placeholder for OpenAI. To add new API providers:

1. Create a new API service class in `src/api_service.py`:

```python
class YourAPIService(ServiceProvider):
    Provider = YourAPIClient  # Your API client library
    API_KEY = os.getenv('YOUR_API_KEY')
    
    def calculate_cost(self, model: str, total_tokens: int) -> float:
        # Implement your provider's pricing logic
        price_per_million = self.get_model_price(model)
        return (total_tokens / 1_000_000) * price_per_million
```

The base `ServiceProvider` class handles the common API call interface, so you only need to implement cost calculation specific to your provider.

Use your new service:

```python
from src.api_service import YourAPIService

api = YourAPIService(TaskData=task)
abc = EnsembleCascade(api, task, cascade_tiers)
```

> **Note**: The OpenAI implementation is currently incomplete — the `calculate_cost` method needs to be implemented with OpenAI's pricing structure.

---

### Adding New Cascade Methods

To implement custom cascade methods, inherit from `CascadeMethod` in `src/methods/base_cascade.py`:

```python
class YourCascadeMethod(CascadeMethod):
    def _inference_cascade(self, prompts: List[str]) -> Tuple[List[str], float]:
        # Implement your custom cascade logic
        # Return predictions and average latency
        pass
```

---


## 🔬 Key Features

- **Training-Free**: ABC requires no additional training  
- **Black-Box Compatible**: Works with any API that returns text responses  
- **Parallel Ensemble Execution**: Tier 1 models run in parallel for speed  
- **Comprehensive Baselines**: Fair comparison with SOTA cascade methods

## 🚨 Important Notes

- **API Costs**: Running full experiments can be expensive! Test with `len_data=10` first  
- **Rate Limits**: Built-in retry mechanism handles temporary API failures  
- **Model Availability**: Ensure all models in cascade tiers are available on Together.ai

---

## 📄 Citation

If you find the work/codebase useful or relevant, please cite:

```bibtex
@article{kolawole2025abc,
  title={Agreement-Based Cascading for Efficient Inference},
  author={Kolawole, Steven and Dennis, Don and Talwalkar, Ameet and Smith, Virginia},
  journal={Transactions on Machine Learning Research},
  year={2025}
  month={7},
  url={https://openreview.net/forum?id=jn9B7LMlzk}
}
```