import os
os.environ['HF_HOME'] = '/data/users/wplacroix/.cache/'
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding, pipeline
import torch
from huggingface_hub import login
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
from tqdm import tqdm


# custom NN model with BERT embeddings

class BERTClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.requires_grad_(False)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.proj_size = 20
        self.hidden_size = 100
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=False, proj_size=self.proj_size)
        self.classifier = torch.nn.Linear(self.proj_size+3, num_classes)

    def forward(self, input_ids, attention_mask, sentiment):
        # dummy forward pass, not real architecture
        outputs = self.bert(input_ids, attention_mask).last_hidden_state
        outputs = self.lstm(outputs)[0][:,-1]
        # insert classification layers here
        # surprisal, sentiment, etc.
        outputs = self.classifier(torch.cat((outputs, sentiment), dim=1))
        return outputs

class CustomDataset(Dataset):
    """
    CustomDataset is a class for creating a dataset in PyTorch, inheriting from the PyTorch Dataset class.
    This dataset is designed to handle tabular data provided as pandas DataFrames.

    Attributes:
        features (pd.DataFrame): A DataFrame containing the features of the dataset.
        labels (pd.Series or pd.DataFrame): A Series or DataFrame containing the labels of the dataset.
    Methods:
        __getitem__(self, index): Returns the features and label for a given index.
        __len__(self): Returns the total number of samples in the dataset.
    """
    def __init__(self, features, labels):
        """
        Parameters:
            features (pd.DataFrame): The features of the dataset.
            labels (pd.Series or pd.DataFrame): The labels of the dataset.
        """
        self.features = pd.DataFrame(features)
        self.labels = pd.DataFrame(labels)

    def __getitem__(self, index):
        """
        Parameters:
            index (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing the features as a numpy array and the label.
        """
        features = self.features.iloc[index].to_numpy()
        label = [self.labels.iloc[index]]
        return features, label

    def __len__(self):
        """
        Returns:
            int: The total number of samples.
        """
        return len(self.features)



def main():

        
    API_TOKEN = "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"
    login(token=API_TOKEN)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize(data):
        return tokenizer(data["statement"], truncation=True, max_length=512, padding=True)  

    batch_size = 32

    dataset = load_dataset("liar")



    distilled_student_sentiment_classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        top_k=None)

    train = dataset["validation"]

    sentiments_list = []

    for statement in tqdm(train["statement"]):
        scores = distilled_student_sentiment_classifier(statement)[0]
        sentiments = [sentiment["score"] for sentiment in scores]
        sentiments_list.append(sentiments)

    print(len(sentiments_list), len(train["statement"]))

    train = train.add_column("sentiment", sentiments_list)

    print(train)


    tokenized_dataset = train.map(tokenize, batch_size=batch_size, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sentiment'])


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    print(next(iter(train_dataloader)).keys())

    # simple training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.CrossEntropyLoss()
    model = BERTClassifier(6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for i in range(100):
        model.train()
        losses = []
        predictions = []
        targets = []
        total = 0
        correct = 0
        for i, batch in tqdm(enumerate(train_dataloader)):
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

            predictions.extend(outputs.detach().argmax(dim=1))
            targets.extend(labels)
            # for sample in zip(batch["labels"], outputs.detach().argmax(dim=1)):
            #     total += 1
            #     if sample[0] == sample[1]:
            #         correct += 1
            batch.to('cpu')
        total = len(targets)
        correct = np.sum(np.array(predictions) == np.array(targets))
        print(correct/total*100, np.mean(losses))
        print(predictions)
    model.to('cpu')
    return


if __name__ == "__main__":

    main()