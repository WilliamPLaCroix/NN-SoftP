import os

import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, LlamaForSequenceClassification, BitsAndBytesConfig, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


#from huggingface_hub import login
#login()

###################
# variables to set
###################

# dataset params
DATASET = "FNHQ/cofacts"
NUM_LABELS = 2

# model params
CHECKPOINT = "meta-llama/Llama-2-7b-hf"

# train params
EPOCHS = 10
BATCH_SIZE = 2
LR = 5e-5
LORA_R = 8
MAX_LENGTH = 500

# logging params
WANDB_PATH = "/data/users/jguertler/.cache/wandb.tok"
WANDB_PROJECT = "llama_clf"


##################
# dataset
##################

dataset = load_dataset(DATASET)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
tokenizer.pad_token=tokenizer.eos_token


def tokenize(batch):
    tokens = tokenizer(batch["text"], max_length=MAX_LENGTH, truncation=True)
    return tokens


tokenized_ds = dataset.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#############
# model
#############

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

model = LlamaForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=NUM_LABELS,
    quantization_config = bnb_config,
    pad_token_id=0
    )

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=LORA_R,
    lora_alpha=32,
    lora_dropout=0.1
    )

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


#############
# logging
#############

with open(WANDB_PATH, "r", encoding="UTF-8") as f:
    wandb_tok = f.read().strip()

wandb.login(key=wandb_tok)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]=WANDB_PROJECT

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


##############
# training
##############

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


training_args = TrainingArguments(
    output_dir="clf",
    report_to="wandb",
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
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()