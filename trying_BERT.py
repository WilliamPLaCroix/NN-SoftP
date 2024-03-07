import os
os.environ['HF_HOME'] = '/data/users/wplacroix/.cache/'
from transformers import AutoModel, DataCollatorWithPadding, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
import numpy as np
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

class Classifier(torch.nn.Module):
    def __init__(self, num_classes, language_model):
        super(Classifier, self).__init__()
        self.requires_grad_(False)
        self.lm = AutoModel.from_pretrained(language_model, quantization_config=bnb_config)
        self.lm_out_size = self.lm.config.hidden_size
        self.proj_size = 20
        self.hidden_size = 100
        self.lstm = torch.nn.LSTM(input_size=self.lm_out_size, hidden_size=self.hidden_size, 
                                  num_layers=2, batch_first=True, bidirectional=False, proj_size=self.proj_size,
                                  dtype=torch.bfloat16)
        self.classifier = torch.nn.Linear(self.proj_size+3, num_classes, dtype=torch.bfloat16)
        #self.classifier = torch.nn.Linear(self.lm_out_size+3, num_classes)
        self.condenser = torch.nn.Linear(self.lm_out_size, self.hidden_size, dtype=torch.bfloat16)
        self.activation = torch.nn.LeakyReLU()
        self.extra_linear_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.bfloat16)
        self.extra_linear_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.bfloat16)
        self.extra_linear_3 = torch.nn.Linear(self.hidden_size, self.proj_size, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, sentiment):
        # dummy forward pass, not real architecture
        outputs = self.lm(input_ids, attention_mask).last_hidden_state
        print("lm output", outputs.shape, outputs.dtype)
        # outputs = self.lstm(outputs)[0][:,-1]
        outputs = torch.mean(outputs, dim=1, dtype=torch.bfloat16)
        print("mean output", outputs.shape, outputs.dtype)
        outputs = self.condenser(outputs)
        print("condensed output", outputs.shape, outputs.dtype)
        outputs = self.activation(outputs)
        print("activation output", outputs.shape, outputs.dtype)
        outputs = self.extra_linear_1(outputs)
        print("linear 1 output", outputs.shape, outputs.dtype)
        outputs = self.activation(outputs)
        print("activation output", outputs.shape, outputs.dtype)
        outputs = self.extra_linear_2(outputs)
        print("linaer 2 output", outputs.shape, outputs.dtype)
        outputs = self.activation(outputs)
        print("activation output", outputs.shape, outputs.dtype)
        outputs = self.extra_linear_3(outputs)
        print("linear 3 output", outputs.shape, outputs.dtype)
        outputs = self.activation(outputs)
        print("activation output", outputs.shape, outputs.dtype)
        # insert classification layers here
        # surprisal, sentiment, etc.
        outputs = self.classifier(torch.cat((outputs, sentiment.bfloat16()), dim=1))
        print("classifier output", outputs.shape, outputs.dtype)
        return outputs



def main():

    global bnb_config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    batch_size = 16
    learning_rate = 0.001

    API_TOKEN = "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"
    login(token=API_TOKEN)
    language_model = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(data):
        return tokenizer(data["statement"], truncation=True, max_length=512, padding=True)

    def dataloader_from_pickle(split):
        dataframe = pd.read_pickle(f"./pickle_files/{split}.pkl")
        dataset = Dataset.from_pandas(dataframe)
        tokenized_dataset = dataset.map(tokenize, batch_size=batch_size, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sentiment'])
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)


    train_dataloader = dataloader_from_pickle("train")
    val_dataloader = dataloader_from_pickle("validation")
    test_dataloader = dataloader_from_pickle("test")


    loss_fn = nn.CrossEntropyLoss()
    model = Classifier(6, language_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(10):
        losses = []
        predictions = []
        targets = []
        for batch_number, batch in enumerate(val_dataloader):
            
            batch.to(device)
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            sentiment = batch["sentiment"]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, sentiment)
            print(outputs.shape, labels.shape)
            print(outputs)
            print(labels)
            loss = loss_fn(outputs, labels)
            print(loss.item())
            loss.backward() # this is not working
            optimizer.step()
            losses.append(loss.item())
            predictions.extend(outputs.detach().argmax(dim=1).to('cpu').tolist())
            targets.extend(labels.to('cpu').tolist())
            break
        print("max memory allocated:", torch.cuda.max_memory_allocated())
        print("memory allocated:", torch.cuda.memory_allocated())
        total = len(targets)
        correct = np.sum(np.array(predictions) == np.array(targets))
        print("acc:", correct/total*100, "loss:", np.mean(losses))
        return
    return


if __name__ == "__main__":

    main()