import os
import wget

from datasets import load_dataset, load_metric, Dataset, DatasetDict
import pandas as pd
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


train_df = pd.read_csv("/data/users/jguertler/datasets/liar/train.tsv", sep="\t", usecols=[1,2], names=["label", "text"])
valid_df = pd.read_csv("/data/users/jguertler/datasets/liar/valid.tsv", sep="\t", usecols=[1,2], names=["label", "text"])

label_dict = {
    "true":0,
    "mostly-true":0,
    "half-true":0,
    "barely-true":1,
    "false":1,
    "pants-fire":1
    }

train_df["label"] = [label_dict[lab] for lab in train_df.label]
valid_df["label"] = [label_dict[lab] for lab in valid_df.label]

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", padding=True, truncation=True)

def encode(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

liar_train = Dataset.from_pandas(train_df)
liar_valid = Dataset.from_pandas(valid_df)
liar_train = liar_train.map(encode, batched=True)
liar_valid = liar_valid.map(encode, batched=True)
data_dict = DatasetDict({"train":liar_train, "valid":liar_valid})

roberta_clf = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

train_args = TrainingArguments(
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_tensors=True,
    logging_strategy="epoch"
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    output_dir="./outputs/"
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

   # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted', zero_division=1)
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    roberta_clf,
    train_args,
    train_dataset=data_dict["train"],
    eval_dataset=data_dict["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()