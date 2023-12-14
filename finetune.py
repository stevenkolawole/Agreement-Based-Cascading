import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, pipeline, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score
from datasets import load_dataset
import time
import pandas as pd
import numpy as np

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
train_dataset = dataset['train'].filter(lambda x: x['label'] != 2).shuffle().train_test_split()

models = dict(
    # bert_1 = "robertsamoilescu/movie-sentiment-bert-base-uncased",
    bart_400 = "valhalla/bart-large-sst2",
    bart_140 = "ModelTC/bart-base-sst2",
    bert_110 = "yoshitomo-matsubara/bert-base-uncased-sst2",
    bert_340 = "yoshitomo-matsubara/bert-large-uncased-sst2",
    roberta_110 = "simonycl/roberta-base-sst-2-64-13",
    roberta_340 = "simonycl/roberta-large-sst-2-64-13",
    xlnet_110 = "textattack/xlnet-base-cased-SST-2",
    xlnet_340 = "textattack/xlnet-large-cased-SST-2",
    electra_110 = "howey/electra-base-sst2",
    electra_335 = "howey/electra-large-sst2",
#    bart_140 = "ModelTC/bart-base-sst2",
#    bart_400 = "valhalla/bart-large-sst2",
)

# Function to fine-tune the model on the training dataset
def fine_tune_model(model_name, train_dataset=train_dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    max_seq_length = 64

    def tokenize_batch(batch):
        return tokenizer(batch["text"], padding='max_length', truncation=True, max_length=max_seq_length)

    train_dataset = train_dataset.map(tokenize_batch, batched=True)

    training_args = TrainingArguments(
        output_dir=f"finetuned_models/{model_name}",
        evaluation_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
    )

    def compute_metrics(p):
        try:
            labels = p.predictions.argmax(axis=1)
        except:
            labels = np.argmax(p.predictions[0], axis=1)
        return {"accuracy": accuracy_score(labels, p.label_ids)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return model, tokenizer


predictions = {model_name: [] for model_name in models.keys()}

# Extend the dictionaries to store inference times and accuracy
inference_times = {model_name: [] for model_name in models.keys()}
# accuracies = {model_name: [] for model_name in models.keys()}

test_data = dataset['validation']['text']
predictions["true_labels"] = dataset["validation"]["label"]


for model_key, model_name in models.items():

  print(f"Fine-tuning {model_key}...")
  fine_tuned_model, tokenizer = fine_tune_model(models[model_key])

  # Perform predictions on the test dataset
  pipe = pipeline("sentiment-analysis", model=fine_tuned_model, tokenizer=tokenizer, device=7)
  start_time = time.time()
  result = pipe(test_data)
  if model_key == 'bert_1': # warmup
    continue
  end_time = time.time()
  inference_time = end_time - start_time

  pred = list(map(lambda x: x['label'], result))
  confidence = list(map(lambda x: x['score'], result))
  predictions[model_key] = pred
  predictions[model_key+"_confidence"] = confidence

  inference_times[model_key] = inference_time
  print(f"***********INFERENCE DONE FOR {model_key}")
  print("\n")

  del fine_tuned_model
  del tokenizer
  torch.cuda.empty_cache()


df = pd.DataFrame(predictions)
df = df.loc[df['true_labels'] != 2]

df.to_csv("predictions_finetune.csv")

inf_df = pd.DataFrame([inference_times])
inf_df.to_csv("inference_times_finetune.csv")
