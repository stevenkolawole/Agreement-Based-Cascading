from datasets import load_dataset
from methods.frugalgpt_scorer import Scorer

class Dataset:
    label_column = 'label'
    label_regex = r'Answer:\s*(.*)'
    required_attributes = ['data_url', 'query_column', 'PROMPT_PREFIX']

    def __init__(self, train_frugalgpt_scorer=False):
        for attr in self.required_attributes:
            if not hasattr(self, attr):
                raise ValueError(f"Subclasses must define '{attr} attribute")
        
        self.data = self.load_data(self.data_url)
        self.train_data, self.val_data = self.preprocess()

        with open(self.PROMPT_PREFIX, "r") as f:
            self.base_prompt = f.read()
        
        self._FOLDER = f"scorer_logs/{self.data_url.split('/')[-1]}/"
        if train_frugalgpt_scorer:
            self.process_data_for_training()
            self.train_frugalgpt_scorer()

    def load_data(self, HF_URL):
        return load_dataset(HF_URL)
    
    def preprocess(self):
        return (self.data["train"], self.data["test"])
        
    def process_data_for_training(self): # For FrugalGPT's training
        self.train_data = self._process_data(self.train_data)
        self.val_data = self._process_data(self.val_data)

    def _process_data(self, dataset):
        dataset = dataset.map(self._concatenate_data_columns, remove_columns=dataset.column_names)
        return dataset

    def _concatenate_data_columns(self, example):
        concatenated = f"{example[self.query_column]}\nA: {example[self.label_column]}"
        return {
            "query": concatenated,
            "label": 1 # overwrite the label with quality; quality is 1 since we are using dataset prompts and true labels
        }
    
    def train_frugalgpt_scorer(self):
        self.Scorer = Scorer(TASK_FOLDER=self._FOLDER)
        self.Scorer.pipeline(self.train_data,
                             self.val_data,
                             "query",)
    

class OverrulingDataset(Dataset):
    data_url = "LawInformedAI/overruling"
    query_column = "sentence1"
    label_regex = r'Answer:\s*(\b[01]\b)'
    PROMPT_PREFIX = "src/prompt_templates/overruling.txt"

    def preprocess(self):
        data_split = self.data['train'].train_test_split(test_size=.4)
        return (data_split['train'], data_split['test'])


class HeadlineDataset(Dataset):
    data_url = "steve1989/financial_news_headlines"
    query_column = "Headlines"
    label_column = "sentiment_label"
    label_regex = r'Answer:\s*(\b(?:up|down|neutral|none)\b)'
    PROMPT_PREFIX = "src/prompt_templates/headlines.txt"


class CoQADataset(Dataset):
    data_url = "stanfordnlp/coqa"
    query_column = "context_and_question"
    label_column = "answer"
    PROMPT_PREFIX = "src/prompt_templates/coqa.txt"

    def add_story_with_first_qa(self, example):
        story = example['story']
        first_question = example['questions'][0]
        example[self.label_column] = example['answers']['input_text'][0]
        example[self.query_column] = f"{story}\nQuestion: {first_question}"
        return example

    def preprocess(self):
        train_data = self.data["train"].map(self.add_story_with_first_qa)
        val_data = self.data["validation"].map(self.add_story_with_first_qa)
        return (train_data, val_data)


class GSM8KDataset(Dataset):
    data_url = "openai/gsm8k"
    query_column = "question"
    label_column = "answer"
    label_regex = r'####\s*(-?\d+(?:\.\d+)?)' # overwrite Dataset's regex
    PROMPT_PREFIX = "src/prompt_templates/gsm8k.txt"

    def load_data(self, HF_URL):
        return load_dataset(HF_URL, "main")


test = OverrulingDataset(train_frugalgpt_scorer=True)
test2 = HeadlineDataset(train_frugalgpt_scorer=True)
test3 = CoQADataset(train_frugalgpt_scorer=True)
test4 = GSM8KDataset(train_frugalgpt_scorer=True)
