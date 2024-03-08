from datasets import load_dataset
from transformers import TrainingArguments, Trainer, XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast, DataCollatorWithPadding
import evaluate
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from huggingface_hub import login

TOK_PATH = "/data/users/jguertler/.cache/token"
#TOK_PATH = "/home/wolfingten/.cache/huggingface/token"

with open(TOK_PATH, "r", encoding="utf8") as f:
    token = f.read().strip()

login(token)

EPOCHS = 10
BATCH_SIZE = 8
LR = 2e-5
MAX_LENGTH = 512

CHECKPOINT = "xlm-roberta-base"

DATASET = "liar"
NUM_LABELS = 6

dataset = load_dataset(DATASET)

tokenizer = XLMRobertaTokenizerFast.from_pretrained(CHECKPOINT, padding=True, truncation=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.model_max_len=MAX_LENGTH


def tokenize(batch):
    return tokenizer(batch["statement"], padding="longest", truncation=True, max_length=MAX_LENGTH)

tokenized_ds = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = XLMRobertaForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=NUM_LABELS,
    classifier_dropout=0.1)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


training_args = TrainingArguments(
    output_dir="clf",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
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