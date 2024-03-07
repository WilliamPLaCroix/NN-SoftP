import sys
from typing import List, Any, Dict
from datasets import load_dataset
from transformers.data import  *
from transformers import AutoTokenizer, TrainingArguments, Trainer, LlamaForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np
import torch
from huggingface_hub import login


login()

epochs = 10
batch_size = 8
learning_rate = 5e-5
lora_r = 12
max_length = 64

checkpoint = "meta-llama/Llama-2-7b-hf"

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
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
model = LlamaForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=num_labels,
    id2label=id2lab,
    label2id=lab2id,
    quantization_config = bnb_config
    )


peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=lora_r,
    lora_alpha=32,
    lora_dropout=0.1
    )
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def tokenize(batch):
    return tokenizer(batch, padding='longest', max_length=max_length, truncation=True)


tokenized_ds = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
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
    compute_metrics=compute_metrics,
)

trainer.train()