import time

import pandas as pd


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from datasets import load_dataset


################################################
# LOAD LIAR DATASET & SPLIT & TURN INTO PANDAS #
################################################

raw_liar_dataset = load_dataset("liar")
raw_liar_dataset_train = raw_liar_dataset["train"]
raw_liar_dataset_validation = raw_liar_dataset["validation"]
raw_liar_dataset_test = raw_liar_dataset["test"]

columns_to_keep = ["statement", "label"]

pd_liar_train = pd.DataFrame(raw_liar_dataset_train)
pd_liar_train = pd_liar_train[columns_to_keep]

####
pd_liar_train = pd_liar_train.head(200)
####

pd_liar_validation = pd.DataFrame(raw_liar_dataset_validation)
pd_liar_validation = pd_liar_validation[columns_to_keep]

pd_liar_test = pd.DataFrame(raw_liar_dataset_test)
pd_liar_test = pd_liar_test[columns_to_keep]


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
"""
def pad_collate(batch):
    # Find maximum sequence length in the batch
    # print(f"batch is: {batch}")
    for data in batch:
        """
        #print(f"data[0] in batch is: {data[0]}")
        #print(f"data[1] in batch is: {data[1]}")
"""
    max_length = max([len(tokenizer.encode(data[0])) for data in batch])
    # print(f"max_length of batch is: {max_length}")
    # Pad all sequences to the same length
    padded_batch = []
    for data, label in batch:
        tokenized_data = tokenizer(data, return_tensors="pt")
        print()
        print(f"TOKENIZED DATA IS: {tokenized_data}")
        print(f"TOKENIZED DATA[0] IS: {tokenized_data[0]}")
        print(f"len of TOKENIZED DATA[0] IS: {len(tokenized_data)}")
        print(f"TYPE OF TOKENIZED DATA ['input_ids']: {type(tokenized_data['input_ids'])}")
        print(f"SHAPE OF TOKENIZED DATA ['input_ids'] {tokenized_data['input_ids'].shape}")
        print()
        #print(f"TOKENIZED DATA IS: {tokenized_data}")
        #print(f"TYPE OF TOKENIZED DATA IS: {type(tokenized_data)} AND LENGTH IS: {len(tokenized_data)}")
        #print(f"tokenized_data[0] IS: {tokenized_data[0]}")
        #print()
        #padded_data = tokenized_data[0].pad(max_length - len(tokenized_data))
        padding_tensor = torch.zeros(1, max_length - len(tokenized_data["input_ids"]))
        #padded_data = tokenized_data["input_ids"] + [0] * (max_length - len(tokenized_data["input_ids"]))
        padded_data = torch.cat((tokenized_data["input_ids"], padding_tensor), dim=1)
        print()
        print(f"PADDED DATA IS: {padded_data}")
        print()
        padded_batch.append((padded_data, label))
    return padded_batch
"""
train_dataset = LiarDataset(pd_liar_train)
#train_dataset = train_dataset.select(range(100))
#train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

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
        return F.softmax(logits, dim=-1)


#####################
# LOAD LLAMA2 MODEL #
#####################

PATH = "/home/pj/Schreibtisch/LLAMA/LLAMA_hf/"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
model = AutoModel.from_pretrained(PATH, local_files_only=True, quantization_config=bnb_config)
classifier = LLama2ClassificationHead(6)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.01)


####
###
#### FOR BATCH SIZE = 1
####
# Training loop
num_epochs = 3
training_start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    for inputs, labels in train_dataloader:

        inputs = tokenizer(inputs, return_tensors="pt")
        llama_outputs = model(inputs["input_ids"])
        predictions = classifier(llama_outputs[0].float())
        #print()
        #print()
        #print()
        #print(f"PREDICTIONS HAVE TYPE {type(predictions)} WITH DTYPE {predictions.dtype}")
        #print(f"LABELS HAVE TYPE {type(labels)} WITH DTYPE {predictions.dtype}")

        #predictions = predictions.to(torch.long)
        #predictions = predictions.long()
        #labels = labels.long()
        #labels = labels.to(torch.long)
        log_probs = F.log_softmax(predictions, dim=1)
        # Calculate loss
        loss = criterion(log_probs.float(), labels.long())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Time needed: {epoch_end_time - epoch_start_time}')

training_end_time = time.time()
print('Training finished!')
print(f"Full Training time: {training_end_time - training_start_time}")


"""
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_dataloader:
        max_length = max([len(tokenizer.encode(statement)) for statement in inputs])
        tokenized_inputs = torch.tensor(1, max_length)
        for statement in inputs:
            tokenized_statement = tokenizer(statement, return_tensors="pt")

        print(batch[1])
        for inputs, labels in batch:
            print()
            print(f"inputs is: {inputs}\n labels is: {labels}")
            #llama_outputs = model(inputs["input_ids"])[0].float()

            llama_outputs = model(inputs)
            predictions = classifier(llama_outputs)


            # Calculate loss
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
"""
"""
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_dataloader:
        print()
        print()

        print()
        print("BATCH IS")
        print(batch)
        print(len(batch))
        print(batch[0])
        print(f"LEN OF batch[0] IS: {len(batch[0])}")
        print(f"TYPE OF batch[0] IS: {type(batch[0])}")
        print(batch[1])
        for inputs, labels in batch:
            print()
            print(f"inputs is: {inputs}\n labels is: {labels}")
            #llama_outputs = model(inputs["input_ids"])[0].float()

            llama_outputs = model(inputs)
            predictions = classifier(llama_outputs)


            # Calculate loss
            loss = criterion(predictions, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
"""




checkpoint_path = 'first_try.pth'

# Save the model's state dictionary and other relevant information
torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)

print(f"Checkpoint saved at '{checkpoint_path}'")

