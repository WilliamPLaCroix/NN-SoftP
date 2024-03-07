import os
os.environ['HF_HOME'] = '/data/users/wplacroix/.cache/'
from transformers import AutoModel, DataCollatorWithPadding, AutoTokenizer
import torch
from huggingface_hub import login
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

class BERTClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.requires_grad_(False)
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.proj_size = 20
        self.hidden_size = 100
        #self.lstm = torch.nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=False, proj_size=self.proj_size)
        #self.classifier = torch.nn.Linear(self.proj_size+3, num_classes)
        self.classifier = torch.nn.Linear(768+3, num_classes)
        #self.condenser = torch.nn.Linear(768, self.proj_size)

    def forward(self, input_ids, attention_mask, sentiment):
        # dummy forward pass, not real architecture
        outputs = self.bert(input_ids, attention_mask).last_hidden_state
        outputs = torch.mean(outputs, dim=1)
        #outputs = self.condenser(outputs)
        #outputs = self.lstm(outputs)[0][:,-1]
        # insert classification layers here
        # surprisal, sentiment, etc.
        outputs = self.classifier(torch.cat((outputs, sentiment), dim=1))
        return outputs



def main():

    API_TOKEN = "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"
    login(token=API_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32

    def tokenize(data):
        return tokenizer(data["statement"], truncation=True, max_length=512, padding=True)

    def dataloader_from_pickle(split):
        dataframe = pd.read_pickle(f"./picklefiles/{split}.pkl")
        dataset = Dataset.from_pandas(dataframe)
        tokenized_dataset = dataset.map(tokenize, batch_size=batch_size, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sentiment'])
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)


    train_dataloader = dataloader_from_pickle("train")
    val_dataloader = dataloader_from_pickle("validation")
    test_dataloader = dataloader_from_pickle("test")

    loss_fn = nn.CrossEntropyLoss()
    model = BERTClassifier(6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    for i in range(100):
        model.train()
        losses = []
        predictions = []
        targets = []
        total = 0
        correct = 0
        for i, batch in tqdm(enumerate(val_dataloader)):
            batch.to(device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            sentiment = batch["sentiment"]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, sentiment)
            loss = loss_fn(outputs, labels)
            loss.backward() # this is not working
            optimizer.step()
            losses.append(loss.item())
            predictions.extend(outputs.detach().argmax(dim=1).to('cpu').tolist())
            targets.extend(labels.to('cpu').tolist())
        total = len(targets)
        correct = np.sum(np.array(predictions) == np.array(targets))
        print("acc:", correct/total*100, "loss:", np.mean(losses))
    return


if __name__ == "__main__":

    main()