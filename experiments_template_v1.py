import os
import time
import copy
import pandas as pd
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset
from huggingface_hub import login
import accelerate
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, BitsAndBytesConfig

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
# access token

# "KEEP_COLUMNS" : ["statement", "label"]
# QUANTIZATION
# LM
# IMPLEMENTATION
# Optimizer

# CLF_Head
# device
# experiments output file



"""
######################################################################################################################
___________                           .__                       __           _________                _____.__
\_   _____/__  _________   ___________|__| _____   ____   _____/  |_  ______ \_   ___ \  ____   _____/ ____\__| ____
 |    __)_\  \/  /\____ \_/ __ \_  __ \  |/     \_/ __ \ /    \   __\/  ___/ /    \  \/ /  _ \ /    \   __\|  |/ ___\
 |        \>    < |  |_> >  ___/|  | \/  |  Y Y  \  ___/|   |  \  |  \___ \  \     \___(  <_> )   |  \  |  |  / /_/  >
/_______  /__/\_ \|   __/ \___  >__|  |__|__|_|  /\___  >___|  /__| /____  >  \______  /\____/|___|  /__|  |__\___  /
        \/      \/|__|        \/               \/     \/     \/          \/          \/            \/        /_____/
######################################################################################################################

"LM" :      "Llama-2-7b":
            "Llama-2-13b":
            "Llama-2-70b":

            "Gemma-2b":
            "Gemma-7b":

            "BERT":
            "ROBERTA":
            "XLM-ROBERTA":

            "Mixtral-8x7B":


"HUGGINGFACE_IMPLEMENTATION" :      "Basic"
                                    "CausalLM"
                                    "SequenceClassification"


"CLF_HEAD" :    "SimplestLinearHead"


"FREEZE_LM" :   True
                False


"BATCH_SIZE" :  some integer value


"NUM_EPOCHS" :  some integer value


"EARLY_STOPPING_AFTER" :    some integer value -- determines when to stop training after no validation accuracy improvement has accured


"LEARNING_RATE" :   some float value


"OPTIMIZER" :   "Adam"


"QUANTIZATION" :    True
                    False - not implemented


"DATASET" :     "Liar"
                No other dataset implemented so far


"DATA_FRAC" :   The fraction of the dataset to be used


"KEEP_COLUMNS" :    "ALL"
                    ["statement", "label"]


"NUM_CLASSES" : 6,


"LABEL_MAPPING" : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5},


"EXPERIMENT_NAME" : f"LLAMA2-7b_test1_{time.time()}"

"""

experiment_config_1 = {
    "LM" : "Llama-2-7b",
    "HUGGINGFACE_IMPLEMENTATION" : "Basic",
    "CLF_HEAD" : "SimplestLinearHead",
    "FREEZE_LM" : True,
    "BATCH_SIZE" : 1,
    "NUM_EPOCHS" : 5,
    "EARLY_STOPPING_AFTER" : 5,
    "LEARNING_RATE" : 0.001,
    "OPTIMIZER" : "Adam",
    "QUANTIZATION" : True,
    "DATASET" : "Liar",
    "DATA_FRAC" : 0.001,
    "KEEP_COLUMNS" : ["statement", "label"],
    "NUM_CLASSES" : 6,
    "LABEL_MAPPING" : {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5},
    "EXPERIMENT_NAME" : f"LLAMA2-7b_test1_{time.time()}"

    }

experiment_config_2 = {

    }

experiment_config_3 = {

    }

experiment_config_4 = {

    }

experiments_list = list()
experiments_list.append(experiment_config_1)
experiments_list.append(experiment_config_2)
experiments_list.append(experiment_config_3)
experiments_list.append(experiment_config_4)



################################################
# DATASETS #
################################################

def prepare_dataset (name:str, frac:float, columns:list[str]) -> (object, object, object):
    """
    TODO: Implement other datasets
    TODO: Docstring
    """
    if name == "Liar":
        # load from csv as is in github
        #raw_liar_dataset_train = pd.read_csv("pickle_files/liar_train.csv")
        #raw_liar_dataset_validation = pd.read_csv("pickle_files/liar_val.csv")
        #raw_liar_dataset_test = pd.read_csv("pickle_files/liar_test.csv")

        raw_liar_dataset_train = pd.read_csv("liar_train.csv")
        raw_liar_dataset_validation = pd.read_csv("liar_val.csv")
        raw_liar_dataset_test = pd.read_csv("liar_test.csv")

        # convert into pandas dataframe
        train = pd.DataFrame(raw_liar_dataset_train)
        validation = pd.DataFrame(raw_liar_dataset_validation)
        test = pd.DataFrame(raw_liar_dataset_test)


    def take_top_n_rows (frac:float, train_set:object, val_set:object, test_set:object) -> (object, object, object):
        """
        """
        # determine size
        train_size = int(len(train_set) * frac)
        val_size = int(len(val_set) * frac)
        test_size = int(len(test_set) * frac)
        # apply shrinkage
        train_set = train_set.head(train_size)
        validation_set = val_set.head(val_size)
        test_set = test_set.head(test_size)

        return train_set, validation_set, test_set


    if frac < 1.0:
        train, validation, test = take_top_n_rows(frac, train, validation, test)

    if columns != "ALL":
        train = train[columns]
        validation = validation[columns]
        test = test[columns]

    return train, validation, test




"""
   TODO:      adapt and make it good

def dataloader(datasplit, ):
    tokenized_dataset = dataset.map(tokenize, batch_size=batch_size, batched=True)
        global number_of_labels
        number_of_labels = len(set(tokenized_dataset["label"]))
        global class_weights
        class_weights = torch.tensor(pd.Series(tokenized_dataset["label"]).value_counts(normalize=True, ascending=True),
                                     dtype=bnb_config.bnb_4bit_compute_dtype).to(device)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'sentiment', 'perplexity'])
        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
"""



################################################
# CLASSIFICATION HEADS #
################################################



class SimplestLinearHead(nn.Module):
    def __init__(self, lm_output_size:int, num_classes:int):
        super(SimplestLinearHead, self).__init__()
        self.fc = nn.Linear(lm_output_size, num_classes)

    def forward(self, x):
        pooled_output = torch.mean(x, dim=1)
        logits = self.fc(pooled_output)
        return logits


"""
        #### TODO: Add CNN Head
            TODO: Adapt Multiple Linear Head


"""



class Classifier(torch.nn.Module):
    def __init__(self, language_model):
        super(Classifier, self).__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(language_model, quantization_config=bnb_config)#, device_map='auto')
        for param in self.lm.base_model.parameters():
            param.requires_grad = False
        self.lm_out_size = self.lm.config.hidden_size
        self.proj_size = 40
        self.intermediate_size = 400
        self.hidden_size = 100
        #self.lstm = torch.nn.LSTM(input_size=self.lm_out_size, hidden_size=self.hidden_size,
                                  #num_layers=1, batch_first=True, bidirectional=False, dtype=bnb_config.bnb_4bit_compute_dtype)#, proj_size=self.proj_size,)
        #self.lstm_classifier = torch.nn.Linear(self.hidden_size+4, num_classes, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.activation = torch.nn.LeakyReLU()
        self.batch_norm = torch.nn.BatchNorm1d(self.lm_out_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.condenser_1 = torch.nn.Linear(self.lm_out_size, self.intermediate_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.condenser_2 = torch.nn.Linear(self.intermediate_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_1 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_2 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        # self.extra_linear_3 = torch.nn.Linear(self.hidden_size, self.hidden_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.reducer = torch.nn.Linear(self.lm_out_size+4, self.intermediate_size, dtype=bnb_config.bnb_4bit_compute_dtype)
        self.classifier = torch.nn.Linear(self.intermediate_size, number_of_labels, dtype=bnb_config.bnb_4bit_compute_dtype)


    def forward(self, input_ids, attention_mask, sentiment, perplexity):
        lm_out = self.lm(input_ids, attention_mask, output_hidden_states=True)
        outputs = lm_out.hidden_states[-1]
        #outputs = self.lstm(outputs)[0][:,-1]
        #logits = torch.nn.functional.softmax(lm_out.logits, dim=-1)
        # probs = torch.gather(logits, dim=2, index=input_ids.unsqueeze(dim=2)).squeeze(-1)
        # subword_surp = -1 * torch.log2(probs) * attention_mask
        # mean_surprisal = subword_surp.sum(dim=1) / attention_mask.sum(dim=1)
        outputs = torch.mean(outputs, dim=1, dtype=bnb_config.bnb_4bit_compute_dtype)
        outputs = self.batch_norm(outputs)
        outputs = torch.cat((outputs,
                                    sentiment.to(bnb_config.bnb_4bit_compute_dtype),
                                    perplexity.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)),
                                    # mean_surprisal.to(bnb_config.bnb_4bit_compute_dtype).unsqueeze(-1)),
                                dim=1)

        # outputs = self.batch_norm(outputs)
        # outputs = self.condenser_1(outputs)
        # outputs = self.activation(outputs)
        outputs = self.reducer(outputs)
        outputs = self.activation(outputs)
        outputs = self.classifier(outputs)
        return outputs






################################################
# CLASSIFICATION HEADS #
################################################


def run_experiments (experiments_list:list[dict]) -> None:

    for i, experiment in enumerate(experiments_list):
        if len(experiment) == 0:
            continue

        print(f"Starting experiment no.{i} with configuration:")
        print(experiment)
        output_log_string = "" + str(experiment) + "\n"


        #####################################################################################
        # 1.) Prepare dataset
        train, validation, test = prepare_dataset(experiment["DATASET"], experiment["DATA_FRAC"], experiment["KEEP_COLUMNS"])

        #####################################################################################
        # 2.) Pre-trained stuff /// Huggingface repository
        # 2.1.1.) Llamas
        if experiment["LM"] == "Llama-2-7b":
            huggingface_repo = "meta-llama/Llama-2-7b-hf"
        elif experiment["LM"] == "Llama-2-13b":
            huggingface_repo = "meta-llama/Llama-2-13b-hf"
        elif experiment["LM"] == "Llama-2-70b":
            huggingface_repo = "meta-llama/Llama-2-70b-hf"
        # 2.1.2.) Gemmas
        elif experiment["LM"] == "Gemma-2b":
            huggingface_repo = "google/gemma-2b"
        elif experiment["LM"] == "Gemma-7b":
            huggingface_repo = "google/gemma-7b"
        # 2.1.3.) BERTs
        elif experiment["LM"] == "BERT":
            huggingface_repo = "google-bert/bert-base-uncased"
        # 2.1.4.) ROBERTAs
        elif experiment["LM"] == "ROBERTA":
            huggingface_repo = "FacebookAI/roberta-base"
        elif experiment["LM"] == "XLM-ROBERTA":
            huggingface_repo = "FacebookAI/xlm-roberta-base"
        # 2.1.5.) MIXTRAL
        elif experiment["LM"] == "Mixtral-8x7B":
            huggingface_repo = "mistralai/Mixtral-8x7B-v0.1"

        #####################################################################################
        # 2.2.) Load tokenizer
        PATH = "/home/pj/Schreibtisch/LLAMA/LLAMA_hf/"
        tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
        #tokenizer = AutoTokenizer.from_pretrained(huggingface_repo, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def tokenize(data):
            label_mapping = experiment["LABEL_MAPPING"]
            tokens = tokenizer(data["statement"])
            binary_labels = [label_mapping[label] for label in data["label"]]
            tokens["label"] = binary_labels
            return tokens

        def dataloader(datasplit, batch_size, columns_to_keep):
            dataset = Dataset.from_pandas(datasplit)
            tokenized_dataset = dataset.map(tokenize, batch_size=batch_size, batched=True)
            global number_of_labels
            number_of_labels = len(set(tokenized_dataset["label"]))
            global class_weights
            #class_weights = torch.tensor(pd.Series(tokenized_dataset["label"]).value_counts(normalize=True, ascending=True), dtype=bnb_config.bnb_4bit_compute_dtype).to(device)
            tokenized_dataset.set_format(type='torch', columns=["input_ids", "label"])
            return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

        train_dataloader = dataloader(train, 4, experiment["KEEP_COLUMNS"])
        val_dataloader = dataloader(validation, 4, experiment["KEEP_COLUMNS"])

        # 2.3.) Load LM
        if experiment["HUGGINGFACE_IMPLEMENTATION"] == "Basic":

            lm = AutoModel.from_pretrained(PATH, local_files_only=True, quantization_config=bnb_config)
            #lm = AutoModel.from_pretrained(huggingface_repo, token=access_token, quantization_config=bnb_config)

        elif experiment["HUGGINGFACE_IMPLEMENTATION"] == "CausalLM":
            lm = AutoModelForCausalLM.from_pretrained(huggingface_repo, token=access_token, quantization_config=bnb_config)
        elif experiment["HUGGINGFACE_IMPLEMENTATION"] == "SequenceClassification":
            lm = AutoModelForSequenceClassification.from_pretrained(huggingface_repo, token=access_token, quantization_config=bnb_config)

        # 2.4.) Freeze or not
        if experiment["FREEZE_LM"]:
            if experiment["HUGGINGFACE_IMPLEMENTATION"] == "Basic":
                for param in lm.parameters():
                    param.requires_grad = False
            else:
                for param in lm.base_model.parameters():
                    param.requires_grad = False

        #####################################################################################
        # 3.) Classification head
        if experiment["CLF_HEAD"] == "SimplestLinearHead":
            classifier = SimplestLinearHead(lm.config.hidden_size, experiment["NUM_CLASSES"])

        #####################################################################################
        # 4.) Loss function
        loss_fn = nn.CrossEntropyLoss()

        #####################################################################################
        # 5.) Optimizer
        if experiment["OPTIMIZER"] == "Adam":
            optimizer = optim.Adam(classifier.parameters(), lr=experiment["LEARNING_RATE"])

        #####################################################################################
        # 6.) Training loop
        training_start_time = time.time()
        epoch_times_list = []
        epoch_train_loss_list = []
        epoch_train_acc_list = []
        epoch_val_loss_list = []
        epoch_val_acc_list = []

        # Variables for early stopping
        best_val_acc_so_far = 0
        epochs_without_improvement_counter = 0

        #####################################################################################
        # 6.1.) Epochs
        for epoch in range(experiment["NUM_EPOCHS"]):
            epoch_start_time = time.time()
            classifier.train()

            losses, predictions, targets = [], [], []

            #####################################################################################
            # 6.2.) Training
            for batch_number, batch in enumerate(train_dataloader):
                #batch.to(device)
                optimizer.zero_grad()


                #### TODO: Adjust functionality for more inputs
                #outputs = lm(batch["input_ids"], batch["attention_mask"], batch["sentiment"], batch["perplexity"])

                lm_outputs = lm(batch["input_ids"])
                classifier_outputs = classifier(lm_outputs[0].float())

                loss = loss_fn(classifier_outputs, batch["labels"])
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                predictions.extend(classifier_outputs.detach().argmax(dim=1).to('cpu').tolist())
                targets.extend(batch["labels"].to('cpu').tolist())

            #####################################################################################
            # 6.3.) Epoch training output, printing, metrics
            epoch_string = f"Epoch [{epoch+1}/{experiment['NUM_EPOCHS']}]:"
            memory_info = "Max memory allocated: " + str(torch.cuda.max_memory_allocated()) + "; Memory allocated: " + str(torch.cuda.memory_allocated())

            total = len(targets)
            correct = np.sum(np.array(predictions) == np.array(targets))
            mean_loss = np.mean(losses)
            train_acc = correct/total
            loss_and_accuracy = "Mean train loss: " + str(np.mean(losses)) + "; Train acc: " + str(correct/total*100) + "%"

            print(epoch_string)
            print(memory_info)
            print(loss_and_accuracy)

            epoch_train_loss_list.append(mean_loss)
            epoch_train_acc_list.append(train_acc)

            #####################################################################################
            # 6.4.) Validation
            classifier.eval()
            with torch.no_grad():

                losses, predictions, targets = [], [], []

                for batch_number, batch in enumerate(val_dataloader):
                    #batch.to(device)

                    #outputs = model(batch["input_ids"], batch["attention_mask"], batch["sentiment"], batch["perplexity"])

                    lm_outputs = lm(batch["input_ids"])
                    classifier_outputs = classifier(lm_outputs[0].float())

                    loss = loss_fn(classifier_outputs, batch["labels"])
                    losses.append(loss.item())
                    predictions.extend(classifier_outputs.detach().argmax(dim=1).to('cpu').tolist())
                    targets.extend(batch["labels"].to('cpu').tolist())
                total = len(targets)
                correct = np.sum(np.array(predictions) == np.array(targets))
                print("val loss:", np.mean(losses), "val acc:", accuracy_score(targets, predictions)*100, "val f1:",
                  "val conf:\n", confusion_matrix(targets, predictions)) #f1_score(targets, predictions)*100,

            epoch_val_acc_list.append(accuracy_score(targets, predictions))
            epoch_val_loss_list.append(np.mean(losses))


            epoch_end_time = time.time()
            epoch_time_elapsed = epoch_end_time - epoch_start_time
            epoch_times_list.append(epoch_time_elapsed)


            # Early
            # early stopping:
            if accuracy_score(targets, predictions) > best_val_acc_so_far:
                best_classifier_so_far = copy.deepcopy(classifier)
                epochs_without_improvement_counter = 0
            else:
                epochs_without_improvement_counter += 1
            if epochs_without_improvement_counter == experiment["EARLY_STOPPING_AFTER"]:
                print(f"Early stopping criterion met. No improvement after {experiment['EARLY_STOPPING_AFTER']} epochs")
                break

        training_end_time = time.time()
        training_time_elapsed = training_end_time - training_start_time
        print(f"Training finished! Training for {i} epochs took: {training_time_elapsed}s")

        output_log_string += "\n" + "Training Loss: \n" + str(epoch_train_loss_list)
        output_log_string += "\n" + "Validation Loss: \n" + str(epoch_val_loss_list)
        output_log_string += "\n" + "Training Acc: \n" + str(epoch_train_acc_list)
        output_log_string += "\n" + "Validation Acc: \n" + str(epoch_val_acc_list)

        #checkpoint_path = 'friday_4_frozen_llama_lr001.pth'

        """
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

    return


run_experiments(experiments_list)

