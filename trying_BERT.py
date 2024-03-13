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
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm



class Classifier(torch.nn.Module):
    def __init__(self, language_model):
        super(Classifier, self).__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(language_model, quantization_config=bnb_config)#, device_map='auto')
        self.requires_grad_(False)
        self.lm_out_size = self.lm.config.hidden_size
        self.hidden_size = 100
        self.activation = torch.nn.LeakyReLU()
        self.reducer = torch.nn.Linear(self.lm_out_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.classifier = torch.nn.Linear(self.hidden_size+5, number_of_labels, dtype=bnb_config.bnb_4bit_compute_dtype)


    def forward(self, input_ids, attention_mask, sentiment, perplexity):
        # lm_out is the foundation of the model output, while hidden_states[-1] gives us word embeddings
        # we do mean pooling to get a single consistently sized vector for the entire sequence
        """
        # TODO : move lm_out and self.lm outside of class declaration
        """
        lm_out = self.lm(input_ids, attention_mask, output_hidden_states=True, labels=input_ids)
        outputs = lm_out.hidden_states[-1]
        outputs = torch.mean(outputs, dim=1, dtype=bnb_config.bnb_4bit_compute_dtype)

        # calculates perplexity as mean subword suprisal from LM output logits
        logits = torch.nn.functional.softmax(lm_out.logits, dim=-1).detach()
        probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(dim=2)).squeeze(-1)
        subword_surp = -1 * torch.log2(probs) * attention_mask
        mean_surprisal = subword_surp.sum(dim=1) / attention_mask.sum(dim=1)

        # bring LM output size down so that it doesn't outweigh the additional features
        outputs = self.reducer(outputs)
        outputs = self.activation(outputs)
        
        # concatenate mean-pooled LM output with the additional features
        outputs = torch.cat((outputs.to(bnb_config.bnb_4bit_compute_dtype), 
                            sentiment.to(bnb_config.bnb_4bit_compute_dtype), 
                            perplexity.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1),
                            mean_surprisal.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)), 
                        dim=1)
        
        # final prediction is reduced to len(class_labels)
        return self.classifier(outputs)


class CNN(nn.Module):
    def __init__(self, language_model):
        super(CNN, self).__init__()
        """
        # TODO : move lm_out and self.lm outside of class declaration
        """
        self.lm = AutoModelForCausalLM.from_pretrained(language_model)#, device_map='auto')
        self.requires_grad_(False)
        self.lm_out_size = self.lm.config.hidden_size

        # keep the rest
        self.out_channels = 128
        self.kernel_size = 5
        self.in_channels = self.lm_out_size + 1  # word embeddings + 1 for surprisal value
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=self.kernel_size)

        # Calculate the size after conv and pooling
        sequence_length = max_sequence_length  # Adjusted based on the input shape 
        conv_seq_length = sequence_length  # kernel_size - 1 for Conv1d
        pooled_seq_length = conv_seq_length // self.kernel_size  # assuming default stride for MaxPool1d

        self.flattened_size = self.out_channels * pooled_seq_length  # 128 is the out_channels from conv1
        self.fc1 = nn.Linear(self.flattened_size, self.flattened_size//2, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.fc2 = nn.Linear(self.flattened_size, number_of_labels, dtype=bnb_config.bnb_4bit_compute_dtype)

    def forward(self, input_ids, attention_mask, sentiment, perplexity):
        lm_out = self.lm(input_ids, attention_mask, output_hidden_states=True, labels=input_ids)
        #print(f"Input shape: {lm_out.shape}")
        outputs = lm_out.hidden_states[-1]
        
        logits = torch.nn.functional.softmax(lm_out.logits, dim=-1).detach()
        word_probabilities = torch.gather(logits, dim=2, index=input_ids.unsqueeze(dim=2)).squeeze(-1)
        subword_surp = -1 * torch.log2(word_probabilities) * attention_mask

        outputs = torch.cat((outputs, 
                            subword_surp.unsqueeze(-1),
                            #sentiment.unsqueeze(-1),
                            ), dim=2)

        outputs = self.conv1(outputs.permute(0,2,1).to(torch.float))
        #print(f"After conv1 shape: {outputs.shape}")
        #outputs = self.relu(outputs)
        outputs = self.pool(outputs).to(bnb_config.bnb_4bit_compute_dtype)
        #print(f"After pooling shape: {outputs.shape}")

        outputs = outputs.view(outputs.size(0), -1)
        #print(f"After flattening shape: {outputs.shape}")
        #outputs = self.fc1(outputs)
        #print(f"After fc1 shape: {outputs.shape}")
        outputs = self.fc2(outputs)
        #print(f"Output shape: {outputs.shape}")
        return outputs


def main():

    TOK_PATH = "/projects/misinfo_sp/.cache/token"

    with open(TOK_PATH, "r", encoding="utf8") as f:
        token = f.read().strip()

    login(token)

    batch_size = 32
    learning_rate = 0.00001
    alpha = 1

    global bnb_config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    language_model = "google/gemma-2b"
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    #tokenizer.pad_token = tokenizer.eos_token
    if language_model == "bert-base-uncased":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def temp_tokenize(data):
        return tokenizer(data["statement"])

    def find_max_length():

        longest_sequence_length = 0
        for split in ["train", "validation", "test"]:
            dataframe = pd.read_pickle(f"./pickle_files/{split}.pkl")
            dataset = Dataset.from_pandas(dataframe)
            tokenized_dataset = dataset.map(temp_tokenize)
            longest = max([len(x["input_ids"]) for x in tokenized_dataset])
            print(f"Longest sequence length in {split}:", longest)
            if longest > longest_sequence_length:
                longest_sequence_length = longest

        return longest_sequence_length

    global max_sequence_length
    max_sequence_length = find_max_length()


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_sequence_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def remap_labels_tokenize(data):
        # tokens = tokenizer(data["statement"])
        # label_mapping = {
        #     0: 0,
        #     1: 1,
        #     2: 2,
        #     3: 3,
        #     4: 4,
        #     5: 5}  # Map positive class labels
        # binary_labels = [label_mapping[label] for label in data["label"]]
        # tokens["label"] = binary_labels
        return tokenizer(data["statement"], padding="max_length", max_length=max_sequence_length)

    # def tokenize(data):
    #     return tokenizer(data["statement"], truncation=True, max_length=512, padding=True)


    def dataloader_from_pickle(split):
        print(f"padding {split} to max length of {max_sequence_length}")
        dataframe = pd.read_pickle(f"./pickle_files/{split}.pkl")
        dataset = Dataset.from_pandas(dataframe)
        tokenized_dataset = dataset.map(remap_labels_tokenize, batch_size=batch_size, batched=True)
        global number_of_labels
        number_of_labels = len(set(tokenized_dataset["label"]))
        dataset_length = len(tokenized_dataset)
        # global max_sequence_length
        # max_sequence_length = len(tokenized_dataset["input_ids"][0])
        # for i in range(10):
        #     print(len(tokenized_dataset["input_ids"][i]))
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
    model = CNN(language_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    

    print(f"training on {device}")
    for epoch in range(1000):
        model.train()
        losses = []
        predictions = []
        targets = []
        for batch_number, batch in tqdm(enumerate(train_dataloader)):
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