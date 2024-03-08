import os
os.environ['HF_HOME'] = '/data/users/wplacroix/.cache/'
from transformers import AutoModel, AutoModelForCausalLM, DataCollatorWithPadding, AutoTokenizer, BitsAndBytesConfig
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
        self.lm = AutoModelForCausalLM.from_pretrained(language_model, quantization_config=bnb_config)#, device_map='auto')
        for param in self.lm.base_model.parameters():
            param.requires_grad = False
        self.lm_out_size = self.lm.config.hidden_size
        self.proj_size = 6
        self.intermediate_size = 10
        self.hidden_size = 100
        #self.lstm = torch.nn.LSTM(input_size=self.lm_out_size, hidden_size=self.hidden_size, 
                                  #num_layers=1, batch_first=True, bidirectional=False, dtype=bnb_config.bnb_4bit_compute_dtype)#, proj_size=self.proj_size,)
        #self.lstm_classifier = torch.nn.Linear(self.hidden_size+4, num_classes, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.activation = torch.nn.Sigmoid()
        # self.batch_norm = torch.nn.BatchNorm1d(self.lm_out_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.condenser_1 = torch.nn.Linear(self.lm_out_size+4, self.intermediate_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.condenser_2 = torch.nn.Linear(self.intermediate_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.reducer = torch.nn.Linear(self.hidden_size, self.proj_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.classifier = torch.nn.Linear(self.lm_out_size+4, num_classes, dtype=bnb_config.bnb_4bit_compute_dtype)


    def forward(self, input_ids, attention_mask, sentiment, perplexity):
        #print("input_ids", input_ids.shape, input_ids.dtype)
        # dummy forward pass, not real architecture
        outputs = self.lm(input_ids, attention_mask, output_hidden_states=True).hidden_states[-1]
        #print(outputs)
        # print("lm output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.lstm(outputs)[0][:,-1]
        outputs = self.classifier(torch.cat((outputs, 
                                    sentiment.to(bnb_config.bnb_4bit_compute_dtype), 
                                    perplexity.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)), 
                                dim=1))
        #outputs = self.activation(outputs)
        #outputs = self.lstm_classifier(torch.cat((outputs, 
                                                #   sentiment.to(bnb_config.bnb_4bit_compute_dtype), 
                                                #   perplexity.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)), 
                                                #   dim=1))
        #outputs = torch.mean(outputs, dim=1, dtype=bnb_config.bnb_4bit_compute_dtype)
        # outputs = self.batch_norm(outputs)
        # print("mean output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.condenser_1(torch.cat((outputs, 
                                            #  sentiment.to(bnb_config.bnb_4bit_compute_dtype), 
                                            #  perplexity.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)), 
                                            #     dim=1))
        # print("condensed output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.activation(outputs)

        #outputs = self.condenser_2(outputs)

        #outputs = self.activation(outputs)
        
        # print("activation output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.extra_linear_1(outputs)
        # print("linear 1 output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.activation(outputs)
        # print("activation output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.extra_linear_2(outputs)
        # print("linaer 2 output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.activation(outputs)
        # print("activation output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.extra_linear_3(outputs)
        # print("linear 3 output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.activation(outputs)
        # print("activation output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.reducer(outputs)
        # print("reducer output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        #outputs = self.activation(outputs)
        # print("activation output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        # insert classification layers here
        # surprisal, sentiment, etc.
        #outputs = self.classifier(outputs)
        # print("classifier output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        # outputs = self.activation(outputs)
        # print("classifier output", outputs.shape, outputs.dtype)
        # print("outputs", outputs)
        return outputs



def main():


    global bnb_config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    batch_size = 32
    learning_rate = 0.01

    API_TOKEN = "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"
    login(token=API_TOKEN)
    language_model = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(data):
        return tokenizer(data["statement"], truncation=True, max_length=512, padding=True)

    def dataloader_from_pickle(split):
        dataframe = pd.read_pickle(f"./pickle_files/{split}.pkl")
        dataset = Dataset.from_pandas(dataframe)
        tokenized_dataset = dataset.map(tokenize, batch_size=batch_size, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sentiment', 'perplexity'])
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)


    train_dataloader = dataloader_from_pickle("train")
    val_dataloader = dataloader_from_pickle("validation")
    test_dataloader = dataloader_from_pickle("test")


    loss_fn = nn.CrossEntropyLoss()
    model = Classifier(6, language_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    

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
        print("max memory allocated:", torch.cuda.max_memory_allocated())
        print("memory allocated:", torch.cuda.memory_allocated())
        total = len(targets)
        correct = np.sum(np.array(predictions) == np.array(targets))
        print("train acc:", correct/total*100, "train loss:", np.mean(losses))

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
            print("val acc:", correct/total*100, "val loss:", np.mean(losses))
    return


if __name__ == "__main__":

    main()