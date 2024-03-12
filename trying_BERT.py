import os
os.environ['HF_HOME'] = '/data/users/wplacroix/.cache/'
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



class Classifier(torch.nn.Module):
    def __init__(self, language_model):
        super(Classifier, self).__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(language_model, quantization_config=bnb_config)#, device_map='auto')
        self.requires_grad_(False)
        self.lm_out_size = self.lm.config.hidden_size
        self.proj_size = 40
        self.intermediate_size = 100
        self.hidden_size = 100
        #self.lstm = torch.nn.LSTM(input_size=self.lm_out_size, hidden_size=self.hidden_size, 
                                  #num_layers=1, batch_first=True, bidirectional=False, dtype=bnb_config.bnb_4bit_compute_dtype)#, proj_size=self.proj_size,)
        #self.lstm_classifier = torch.nn.Linear(self.hidden_size+4, num_classes, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.activation = torch.nn.Sigmoid()
        self.batch_norm = torch.nn.BatchNorm1d(self.lm_out_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.condenser_1 = torch.nn.Linear(self.lm_out_size, self.intermediate_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.condenser_2 = torch.nn.Linear(self.intermediate_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.reducer = torch.nn.Linear(self.lm_out_size+5, self.intermediate_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.classifier = torch.nn.Linear(self.intermediate_size, number_of_labels, dtype=bnb_config.bnb_4bit_compute_dtype)


    def forward(self, input_ids, attention_mask, sentiment, perplexity):
        lm_out = self.lm(input_ids, attention_mask, output_hidden_states=True, labels=input_ids)
        outputs = lm_out.hidden_states[-1]
        #outputs = self.lstm(outputs)[0][:,-1]
        logits = torch.nn.functional.softmax(lm_out.logits, dim=-1)
        probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(dim=2)).squeeze(-1)
        subword_surp = -1 * torch.log2(probs) * attention_mask
        mean_surprisal = subword_surp.sum(dim=1) / attention_mask.sum(dim=1)
        outputs = torch.mean(outputs, dim=1, dtype=bnb_config.bnb_4bit_compute_dtype)
        #outputs = self.batch_norm(outputs)
        outputs = torch.cat((outputs, 
                                    sentiment.to(bnb_config.bnb_4bit_compute_dtype), 
                                    perplexity.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1),
                                    mean_surprisal.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)), 
                                dim=1)

        # outputs = self.batch_norm(outputs)
        # outputs = self.condenser_1(outputs)
        # outputs = self.activation(outputs)
        outputs = self.reducer(outputs)
        outputs = self.activation(outputs)
        outputs = self.classifier(outputs)
        print(outputs)
        return outputs



def main():

    TOK_PATH = "/projects/misinfo_sp/.cache/token"

    with open(TOK_PATH, "r", encoding="utf8") as f:
        token = f.read().strip()

    login(token)

    batch_size = 32
    learning_rate = 0.1
    alpha = 1

    global bnb_config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )

    language_model = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def tokenize(data):
    #     return tokenizer(data["statement"], truncation=True, max_length=512, padding=True)
    def tokenize(data):
        tokens = tokenizer(data["statement"])
        label_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5}  # Map positive class labels
        binary_labels = [label_mapping[label] for label in data["label"]]
        tokens["label"] = binary_labels
        return tokens

    def dataloader_from_pickle(split):
        dataframe = pd.read_pickle(f"./pickle_files/{split}.pkl")
        dataset = Dataset.from_pandas(dataframe)
        tokenized_dataset = dataset.map(tokenize, batch_size=batch_size, batched=True)
        global number_of_labels
        number_of_labels = len(set(tokenized_dataset["label"]))
        dataset_length = len(tokenized_dataset)
        weights = torch.as_tensor(pd.Series([dataset_length for _ in range(number_of_labels)]), dtype=bnb_config.bnb_4bit_compute_dtype)
        class_proportions = torch.as_tensor(pd.Series(tokenized_dataset["label"]).value_counts(normalize=True, ascending=True), 
                                     dtype=bnb_config.bnb_4bit_compute_dtype)
        global class_weights
        class_weights = weights / class_proportions
        class_weights
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sentiment', 'perplexity'])
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)


    train_dataloader = dataloader_from_pickle("train")
    val_dataloader = dataloader_from_pickle("validation")
    test_dataloader = dataloader_from_pickle("test")

    loss_fn = nn.CrossEntropyLoss()#weight=class_weights.to(device))#*alpha)
    model = Classifier(language_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    print(f"training on {device}")
    for epoch in range(1000):
        model.train()
        losses = []
        predictions = []
        targets = []
        for batch_number, batch in enumerate(train_dataloader):
            batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch["input_ids"], batch["attention_mask"], batch["sentiment"], batch["perplexity"])
            loss = loss_fn(outputs, batch["labels"])
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            predictions.extend(outputs.detach().argmax(dim=1).to('cpu').tolist())
            targets.extend(batch["labels"].to('cpu').tolist())
            # for name, param in model.named_parameters():
            #     try:
            #         print("Model Parameters",name, torch.isfinite(param.grad).all(), "max param.grad", torch.max(abs(param.grad)), "dtype=", param.grad)
            #     except TypeError:
            #         print("Model Parameters",name, "NoneType")
        print("max memory allocated:", torch.cuda.max_memory_allocated())
        print("memory allocated:", torch.cuda.memory_allocated())
        total = len(targets)
        correct = np.sum(np.array(predictions) == np.array(targets))
        print("train loss:", np.mean(losses), "train acc:", correct/total*100)
        # print("train loss:", np.mean(losses), "train acc:", accuracy_score(targets, predictions)*100, "train f1:", 
        #       f1_score(targets, predictions)*100, "train conf:\n", confusion_matrix(targets, predictions))
        

        model.eval()
        with torch.no_grad():
            
            losses = []
            predictions = []
            targets = []
            for batch_number, batch in enumerate(val_dataloader):
                batch.to(device)
                outputs = model(batch["input_ids"], batch["attention_mask"], batch["sentiment"], batch["perplexity"])
                loss = loss_fn(outputs, batch["labels"])
                losses.append(loss.item())
                predictions.extend(outputs.detach().argmax(dim=1).to('cpu').tolist())
                targets.extend(batch["labels"].to('cpu').tolist())
            total = len(targets)
            correct = np.sum(np.array(predictions) == np.array(targets))
            #print("val loss:", np.mean(losses), "val acc:", correct/total*100)
            print("val loss:", np.mean(losses), "val acc:", accuracy_score(targets, predictions)*100, "val f1:", 
                  "val conf:\n", confusion_matrix(targets, predictions)) #f1_score(targets, predictions)*100,

    return

if __name__ == "__main__":

    main()