from transformers import DistilBertTokenizerFast, BertTokenizerFast, AlbertTokenizerFast, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification, AlbertForSequenceClassification
from torch.nn import functional as F

from transformers import set_seed

import re, torch, evaluate, numpy, json


set_seed(2023)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def save_jsonline(data,savepath):
    json_object = json.dumps(str(data), indent=4)
    with open(savepath,'w') as f:
        f.write(json_object)    
    return 


class Scorer(object):
    def __init__(self, TASK_FOLDER, score_type='DistilBert'):
        if(score_type=='DistilBert'):
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        if(score_type=='Bert'):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        if(score_type=='AlBert'):
            self.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
        self.score_type = score_type
        self.TASK_FOLDER = TASK_FOLDER
        
    def train(self,
              dataset,
              query_column,
              score_type='DistilBert',
              ):
        tokenizer = self.tokenizer
        def tokenize_batch(batch):
            return tokenizer(batch[query_column], padding='max_length', truncation=True, max_length=512)
            
        dataset = dataset.map(tokenize_batch, batched=True)

        dataset = dataset.train_test_split(test_size=.6)
        train_dataset, val_dataset = dataset['train'], dataset['test']

        training_args = TrainingArguments(
            output_dir=self.TASK_FOLDER + 'scorer_location',          # output directory
            num_train_epochs=8,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir=self.TASK_FOLDER + 'logs',            # directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",
        	save_strategy ="epoch",
            load_best_model_at_end=True,
            seed=2023,
        )
        score_type = self.score_type
        if(score_type=='DistilBert'):
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        if(score_type=='Bert'):
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        if(score_type=='AlBert'):
            model = AlbertForSequenceClassification.from_pretrained("albert-base-v2")

        model = model.to(device)
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset ,            # evaluation dataset
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        self.trainer = trainer
        self.model = model
        return model
    
    def predict(self, model, text):
        #trainer = self.trainer
        model = self.model
        encoding = self.tokenizer(text, return_tensors="pt",truncation=True, padding=True)
        encoding = {k: v.to(model.device) for k,v in encoding.items()}
        outputs = model(**encoding)
        logit_score = outputs.logits.cpu().detach()
        #return logit_score
        # convert logit score to torch array
        torch_logits = logit_score
        
        # get probabilities using softmax from logit score and convert it to numpy array
        probabilities_scores = F.softmax(torch_logits, dim = -1).numpy()[0]

        return probabilities_scores
    
    def get_model(self):
        return self.model

    def get_score(self,text):
        prob = self.predict("",text)
        return prob[1]

    def gen_score(self,
                  model,
                  texts,
                  ):
        scores = list()
        for text in texts:
            prob = self.predict(model,text)
            scores.append(prob[1])
        return scores
        
    def save(self,savepath):
        self.trainer.save_model(savepath)
        return

    def load(self,loadpath):
        self.model = AutoModelForSequenceClassification.from_pretrained(loadpath)
        return
        
    def save_scores(self,
                    queries,
                    score_path,
                    scores):
        result = dict()
        i = 0
        for item in queries:
            result[item] = scores[i]
            i+=1
        save_jsonline(data=result, savepath=score_path)    
        return
        
    def pipeline(self, train_dataset, val_dataset, ds_column='sentence1'):
        
        # train the model
        model = self.train(train_dataset, query_column=ds_column)
        
        # get scores 
        scores = self.gen_score(model, val_dataset[ds_column])
        self.save_scores(val_dataset[ds_column], self.TASK_FOLDER + "val_scores.json", scores)
        
        scores = self.gen_score(model, train_dataset[ds_column])
        self.save_scores(train_dataset[ds_column], self.TASK_FOLDER + "train_scores.json", scores)
        
        return
    
