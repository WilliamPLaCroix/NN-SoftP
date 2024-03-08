from datasets import load_dataset
from transformers.data import  *
from transformers import AutoTokenizer, TrainingArguments, Trainer, XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast, BitsAndBytesConfig, DataCollatorWithPadding
import evaluate
import numpy as np
import torch
from huggingface_hub import login

login()

epochs = 10
batch_size = 8
learning_rate = 5e-5
lora_r = 12
max_length = 512

checkpoint = "xlm-roberta-base"

dataset = load_dataset("liar")

num_labels = 6

id2lab = {
    "true":0,
    "mostly-true":1,
    "half-true":2,
    "barely-true":3,
    "false":4,
    "pants-fire":5
    }
lab2id = {v: k for k, v in id2lab.items()}


accuracy = evaluate.load("accuracy")

tokenizer = XLMRobertaTokenizerFast.from_pretrained(checkpoint, padding=True, truncation=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.model_max_len=512


def tokenize(batch):
    return tokenizer(batch["statement"], padding="longest", truncation=True, max_length=512)

tokenized_ds = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = XLMRobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


train_args = TrainingArguments(
    output_dir="clf",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()