import time

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_dataset

import pickle

#import matplotlib.pyplot as plt

from huggingface_hub import login
"""
TOK_PATH = "/projects/misinfo_sp/.cache/token"

with open(TOK_PATH, "r", encoding="utf8") as f:
    token = f.read().strip()

login(token)
"""
access_token = "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"

################################################
# LOAD LIAR DATASET & SPLIT & TURN INTO PANDAS #
################################################


#raw_liar_dataset = load_dataset("liar")
#raw_liar_dataset_train = raw_liar_dataset["train"]
#raw_liar_dataset_validation = raw_liar_dataset["validation"]
#raw_liar_dataset_test = raw_liar_dataset["test"]

raw_liar_dataset_train = pd.read_csv("pickle_files/liar_train.csv")
raw_liar_dataset_validation = pd.read_csv("pickle_files/liar_val.csv")
raw_liar_dataset_test = pd.read_csv("pickle_files/liar_test.csv")


# open a file, where you stored the pickled data
#file = open('pickle_files/liar.pkl', 'rb')

# dump information to that file
#raw_liar_dataset = pickle.load(file)

# close the file
#file.close()

#raw_liar_dataset_train = raw_liar_dataset["train"]
#raw_liar_dataset_validation = raw_liar_dataset["validation"]
#raw_liar_dataset_test = raw_liar_dataset["test"]

columns_to_keep = ["statement", "label"]

pd_liar_train = pd.DataFrame(raw_liar_dataset_train)
pd_liar_train = pd_liar_train[columns_to_keep]



pd_liar_validation = pd.DataFrame(raw_liar_dataset_validation)
pd_liar_validation = pd_liar_validation[columns_to_keep]

pd_liar_test = pd.DataFrame(raw_liar_dataset_test)
pd_liar_test = pd_liar_test[columns_to_keep]

######################
# TESTING PARAMETERS #
######################

train_size = 100
val_size = 100
pd_liar_train = pd_liar_train.head(train_size)
pd_liar_validation = pd_liar_validation.head(val_size)


########################
# DATASET & DATALOADER #
########################

class LiarDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        feature = sample["statement"]
        label = torch.tensor(sample['label'], dtype=torch.float16)
        return feature, label

train_dataset = LiarDataset(pd_liar_train)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

val_dataset = LiarDataset(pd_liar_validation)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

##############################
# CUSTOM CLASSIFICATION HEAD #
##############################

class LLama2ClassificationHead(nn.Module):
    def __init__(self, num_classes):

        super(LLama2ClassificationHead, self).__init__()
        self.fc = nn.Linear(4096, num_classes)


    def forward(self, x):
        pooled_output = torch.mean(x, dim=1)
        logits = self.fc(pooled_output)
        return logits


#####################
# LOAD LLAMA2 MODEL #
#####################

#PATH = "/home/pj/Schreibtisch/LLAMA/LLAMA_hf/"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, quantization_config=bnb_config)
print("~FREEZING LLAMA LAYERS~")
for name, param in model.named_parameters():
    param.requires_grad = False
    #print(name, param.requires_grad)
classifier = LLama2ClassificationHead(6)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)


# Training loop
num_epochs = 10
train_total_predictions = 0
train_correct_predictions = 0
train_epochs_accuracy = []
val_total_predictions = 0
val_correct_predictions = 0
val_epochs_accuracy = []

training_start_time = time.time()
epoch_times = []
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    num_epoch_predictions = 0
    num_epoch_correct_predictions = 0
    val_epoch_predictions = 0
    val_epoch_correct_predictions = 0
    epoch_loss = 0
    for inputs, labels in train_dataloader:

        # forward:
        inputs = tokenizer(inputs, return_tensors="pt")
        llama_outputs = model(inputs["input_ids"])
        predictions = classifier(llama_outputs[0].float())

        loss = criterion(predictions, labels.long())


        # backward:
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

        if torch.argmax(predictions) == labels.item():
            train_correct_predictions += 1
            num_epoch_correct_predictions += 1
        num_epoch_predictions += 1
        train_total_predictions += 1

    #Validation:
    for inputs, labels in val_dataloader:
        inputs = tokenizer(inputs, return_tensors="pt")
        llama_outputs = model(inputs["input_ids"])
        predictions = classifier(llama_outputs[0].float())

        if torch.argmax(predictions) == labels.item():
            val_epoch_correct_predictions += 1
        val_epoch_predictions += 1

    current_train_acc = num_epoch_correct_predictions / num_epoch_predictions
    current_val_acc = val_epoch_correct_predictions / val_epoch_predictions
    train_epochs_accuracy.append(current_train_acc)
    val_epochs_accuracy.append(current_val_acc)
    epoch_end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {current_train_acc}, Val Accuracy: {current_val_acc}, Loss: {epoch_loss:.4f}, Time needed: {epoch_end_time - epoch_start_time}')

training_end_time = time.time()
print('Training finished!')
print(f"Full Training time: {training_end_time - training_start_time}")
print()
print()
print(f"NUM OF EPOCHS: {num_epochs}")
print(f"TRAINING ACCURACIES: {train_epochs_accuracy}")
print(f"VAL ACCURACIES: {val_epochs_accuracy}")


"""
checkpoint_path = 'friday_4_frozen_llama_lr001.pth'

# Save the model's state dictionary and other relevant information
torch.save({
            'epoch': num_epochs,
            #'model_state_dict': model.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)

print(f"Checkpoint saved at '{checkpoint_path}'")
"""


"""
xpoints = [i for i in range(num_epochs)]
plt.plot(xpoints, train_epochs_accuracy, c="green")
plt.plot(xpoints, val_epochs_accuracy, c="red")
plt.title("Accuracies")
plt.show()

"""




