import os
os.environ['HF_HOME'] = '/data/users/phawlitschek/.cache/'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

import matplotlib.pyplot as plt




##################################################
EXPERIMENT_NAME = f"aggregation_streamlit_BERT_binary_SimpleLinearHead_{time.time()}"
##################################################
PRINTING_FLAG = True

#### Other experiment details:
"""

"""

####
experiment = {
    "LM" : "BERT", # not used in code, define yourself
    "HUGGINGFACE_IMPLEMENTATION" : "AutoModel", # USED
    "CLF_HEAD" : "SimplestLinearHead", # not used in code, define yourself
    "FREEZE_LM" : True, # USED
    "BATCH_SIZE" : 32, # USED
    "NUM_EPOCHS" : 200, # USED
    "EARLY_STOPPING_AFTER" : "NEVER", # USED
    "LEARNING_RATE" : 0.0001, # USED
    "OPTIMIZER" : "Adam", # not used in code, define yourself
    "QUANTIZATION" : None, # not used in code, define yourself
    "DATASET" : "Liar", # USED
    "DATA_FRAC" : 1, # USED
    "KEEP_COLUMNS" : ["statement", "label"], # USED
    "NUM_CLASSES" : 2, # USED
    "LABEL_MAPPING" : { # USED
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 0,
        5: 0
        },


    }

"""
TOK_PATH = "/projects/misinfo_sp/.cache/token"

with open(TOK_PATH, "r", encoding="utf8") as f:
    token = f.read().strip()

login(token)
"""


access_token = "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"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
"""


#####################################################################################
# Dataset
#####################################################################################


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

        raw_liar_dataset_train = pd.read_csv("/nethome/phawlitschek/NN-SoftP/cluster_templates/liar_train.csv")
        raw_liar_dataset_validation = pd.read_csv("/nethome/phawlitschek/NN-SoftP/cluster_templates/liar_val.csv")
        raw_liar_dataset_test = pd.read_csv("/nethome/phawlitschek/NN-SoftP/cluster_templates/liar_test.csv")

        # convert into pandas dataframe
        train = pd.DataFrame(raw_liar_dataset_train)
        validation = pd.DataFrame(raw_liar_dataset_validation)
        test = pd.DataFrame(raw_liar_dataset_test)

    def take_top_n_rows (frac:float, train:object, val:object, test:object) -> (object, object, object):
        """
        """
        # determine size
        train_size = int(len(train) * frac)
        val_size = int(len(val) * frac)
        test_size = int(len(test) * frac)
        # apply shrinkage
        train = train.head(train_size)
        validation = val.head(val_size)
        test = test.head(test_size)

        return train, validation, test


    if frac < 1.0:
        train, validation, test = take_top_n_rows(frac, train, validation, test)

    if columns != "ALL":
        train = train[columns]
        validation = validation[columns]
        test = test[columns]

    return train, validation, test


def make_new_labels_counting_dict(num_classes:int) -> dict():
    """
    """
    if num_classes == 6:
        classes_dict = {
            0 : 0,
            1 : 0,
            2 : 0,
            3 : 0,
            4 : 0,
            5 : 0
            }

    if num_classes == 2:
        classes_dict = {
            0 : 0,
            1 : 0,
            }

    return classes_dict

def tokenize(data):
    """
    """
    label_mapping = experiment["LABEL_MAPPING"]
    tokens = tokenizer(data["statement"])
    binary_labels = [label_mapping[label] for label in data["label"]]
    tokens["label"] = binary_labels
    return tokens


def dataloader(datasplit, batch_size, columns_to_keep):
    """
    """
    dataset = Dataset.from_pandas(datasplit)
    tokenized_dataset = dataset.map(tokenize, batch_size=batch_size, batched=True)
    global number_of_labels
    number_of_labels = len(set(tokenized_dataset["label"]))
    global class_weights
    #class_weights = torch.tensor(pd.Series(tokenized_dataset["label"]).value_counts(normalize=True, ascending=True), dtype=bnb_config.bnb_4bit_compute_dtype).to(device)
    tokenized_dataset.set_format(type='torch', columns=["input_ids", "label"])
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)



#####################################################################################
# Define classification head here
#####################################################################################

class SimplestLinearHead(nn.Module):
    def __init__(self, lm_output_size:int, num_classes:int):
        super(SimplestLinearHead, self).__init__()
        self.fc = nn.Linear(lm_output_size, num_classes)

    def forward(self, lm_hidden_states):
        pooled_output = torch.max(lm_hidden_states, dim=1)[0]
        logits = self.fc(pooled_output)
        return logits

#####################################################################################
# Running everything defined above
#####################################################################################
#LLAMA_PATH = "/home/pj/Schreibtisch/LLAMA/LLAMA_hf/"

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = '</s>'
#tokenizer.add_special_tokens({'pad_token': '</s>'})

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train, validation, test = prepare_dataset(experiment["DATASET"], experiment["DATA_FRAC"], experiment["KEEP_COLUMNS"])

train_dataloader = dataloader(train, experiment["BATCH_SIZE"], experiment["KEEP_COLUMNS"])
val_dataloader = dataloader(validation, experiment["BATCH_SIZE"], experiment["KEEP_COLUMNS"])
test_dataloader = dataloader(test, experiment["BATCH_SIZE"], experiment["KEEP_COLUMNS"])

#lm = AutoModel.from_pretrained("google-bert/bert-base-uncased", token=access_token, quantization_config=bnb_config)
lm = AutoModel.from_pretrained("google-bert/bert-base-uncased", token=access_token).to(device)
classifier = SimplestLinearHead(lm.config.hidden_size, experiment["NUM_CLASSES"]).to(device)
if PRINTING_FLAG: print(f"Language Model has hidden_size: {lm.config.hidden_size}")

if experiment["FREEZE_LM"]:
    if experiment["HUGGINGFACE_IMPLEMENTATION"] == "AutoModel":
        if PRINTING_FLAG: print("freezing Model... (AutoModel)")
        for param in lm.parameters():
            param.requires_grad = False
    else:
        if PRINTING_FLAG: print(f"freezing Model... For CausalLM or SequenceClassification")
        for param in lm.base_model.parameters():
            param.requires_grad = False

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(classifier.parameters(), lr=experiment["LEARNING_RATE"])

#####################################################################################
# TRAINING LOOP
#####################################################################################

#### TODO: clean up train mean loss list, epoch train loss list, ....










epochs_train_loss_list = []
epochs_train_acc_list = []
# epochs_train_loss_list : list[float] -- stores training mean loss of epochs
# epochs_train_acc_list : list[float] -- stores training accuracy of epochs

epochs_val_loss_list = []
epochs_val_acc_list = []
# epochs_val_loss_list : list[float] -- stores validation mean loss of epochs
# epochs_val_acc_list : list[float] -- stores validation accuracy of epochs

labels_predicted_train_epochs_list = []
true_targets_train_epochs_list = []
correct_predictions_train_epoch_list = []
# labels_predicted_train_epochs_list : list[dict[int]:int] -- stores dictionaries of train labels predicted each epoch
# true_targets_train_epochs_list : list[dict[int]:int] -- stores dictionaries of train true targets each epoch
# correct_predictions_train_epoch : list[dict[int]:int] -- stores dictionaries of train correct predictions each epoch

labels_predicted_val_epochs_list = []
true_targets_val_epochs_list = []
correct_predictions_val_epoch_list = []
# labels_predicted_val_epochs_list : list[dict[int]:int] -- stores dictionaries of val labels predicted each epoch
# true_targets_val_epochs_list : list[dict[int]:int] -- stores dictionaries of val true targets each epoch
# correct_predictions_val_epoch : list[dict[int]:int] -- stores dictionaries of val correct predictions each epoch

best_val_acc_so_far = 0.0
epochs_without_improvement_counter = 0
# best_val_acc_so_far : float -- stores the highest validation accuracy so far (for early stopping)
# epochs_without_improvement_counter : int -- stores the number of epochs without any improvement in validation accuracy (for early stopping)

number_of_epochs_trained = 0
# number_of_epochs_trained : int -- counter of how many epochs have been trained


epochs_times_list = []
# epochs_times_list : list -- stores how much time each epoch took

if PRINTING_FLAG: print(f"Running on device: {device}")
training_start_time = time.time()
try:
    for epoch in range(experiment["NUM_EPOCHS"]):
        epoch_start_time = time.time()

        #####################################################################################
        # 6.2.) Training
        #####################################################################################

        labels_predicted_train_epoch = make_new_labels_counting_dict(experiment["NUM_CLASSES"])
        true_targets_train_epoch = make_new_labels_counting_dict(experiment["NUM_CLASSES"])
        correct_predictions_train_epoch = make_new_labels_counting_dict(experiment["NUM_CLASSES"])
        # labels_predicted_train_epoch : dict[int]:int -- count which labels have been predicted in this training epoch
        # true_targets_train_epoch : dict[int]:int -- count which labels are true in this training epoch
        # correct_predictions_train_epoch : dict[int]:int -- count which predicted labels were correct in this training epoch

        train_losses, train_predictions, train_targets,  = [], [], []
        # train losses -- used to accumulate losses during batch training
        # train_predictions -- used to accumulate predictions during batch training
        # train targets -- used to accumulate true targets during batch training

        classifier.train()

        for batch_number, batch in enumerate(train_dataloader):
            batch.to(device)
            optimizer.zero_grad()

            lm_outputs = lm(batch["input_ids"])
            classifier_outputs = classifier(lm_outputs[0].float())

            loss = loss_fn(classifier_outputs, batch["labels"])
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            train_predictions.extend(classifier_outputs.detach().argmax(dim=1).to('cpu').tolist())
            train_targets.extend(batch["labels"].to('cpu').tolist())

        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)
        num_predictions_train_epoch = len(train_predictions)
        num_predictions_train_epoch_correct = np.sum(train_predictions == train_targets)
        train_accuracy = num_predictions_train_epoch_correct / num_predictions_train_epoch
        epochs_train_acc_list.append(train_accuracy)

        train_mean_loss = np.mean(train_losses)
        epochs_train_loss_list.append(train_mean_loss)



        for target in train_targets:
            true_targets_train_epoch[target] += 1

        for i, prediction in enumerate(train_predictions):
            labels_predicted_train_epoch[prediction] += 1

            if train_predictions[i] == train_targets[i]:
                correct_predictions_train_epoch[prediction] += 1

        labels_predicted_train_epochs_list.append(labels_predicted_train_epoch)
        true_targets_train_epochs_list.append(true_targets_train_epoch)
        correct_predictions_train_epoch_list.append(correct_predictions_train_epoch)


        #####################################################################################
        # 6.4.) Validation
        #####################################################################################
        labels_predicted_val_epoch = make_new_labels_counting_dict(experiment["NUM_CLASSES"])
        true_targets_val_epoch = make_new_labels_counting_dict(experiment["NUM_CLASSES"])
        correct_predictions_val_epoch = make_new_labels_counting_dict(experiment["NUM_CLASSES"])
        # labels_predicted_val_epoch : dict[int]:int -- count which labels have been predicted in this validation epoch
        # true_targets_val_epoch : dict[int]:int -- count which labels are true in this validatio epoch
        # correct_predictions_val_epoch : dict[int]:int -- count which predicted labels were correct in this validation epoch

        classifier.eval()
        with torch.no_grad():
            val_losses, val_predictions, val_targets = [], [], []

            for batch_number, batch in enumerate(val_dataloader):
                batch.to(device)

                #outputs = model(batch["input_ids"], batch["attention_mask"], batch["sentiment"], batch["perplexity"])

                lm_outputs = lm(batch["input_ids"])
                classifier_outputs = classifier(lm_outputs[0].float())

                loss = loss_fn(classifier_outputs, batch["labels"])
                val_losses.append(loss.item())

                val_predictions.extend(classifier_outputs.detach().argmax(dim=1).to('cpu').tolist())
                val_targets.extend(batch["labels"].to('cpu').tolist())


        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        num_predictions_val_epoch = len(val_predictions)
        num_predictions_val_epoch_correct = np.sum(val_predictions == val_targets)
        val_accuracy = num_predictions_val_epoch_correct / num_predictions_val_epoch
        epochs_val_acc_list.append(val_accuracy)

        val_mean_loss = np.mean(val_losses)
        epochs_val_loss_list.append(val_mean_loss)


        for target in val_targets:
            true_targets_val_epoch[target] += 1

        for i, prediction in enumerate(val_predictions):
            labels_predicted_val_epoch[prediction] += 1

            if val_predictions[i] == val_targets[i]:
                correct_predictions_val_epoch[prediction] += 1

        labels_predicted_val_epochs_list.append(labels_predicted_val_epoch)
        true_targets_val_epochs_list.append(true_targets_val_epoch)
        correct_predictions_val_epoch_list.append(correct_predictions_val_epoch)



        epoch_end_time = time.time()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        epochs_times_list.append(epoch_time_elapsed)
        number_of_epochs_trained += 1

        if PRINTING_FLAG:
            print(f"Epoch [{epoch+1}/{experiment['NUM_EPOCHS']}] took {epoch_time_elapsed}s")
            print(f"Train mean loss: {train_mean_loss}, train accuracy: {train_accuracy}")
            print(f"Val mean loss: {val_mean_loss}, val accuracy: {val_accuracy}")
            print()
            print("TRAINING:")
            print(f"Labels predicted: \t True targets: \t Correct labels:")
            for label in labels_predicted_train_epoch:
                print(f"{labels_predicted_train_epoch[label]} \t\t\t {true_targets_train_epoch[label]} \t\t\t {correct_predictions_train_epoch[label]}")
            print("VALIDATION:")
            print(f"Labels predicted: \t True targets: \t Correct labels:")
            for label in labels_predicted_val_epoch:
                print(f"{labels_predicted_val_epoch[label]} \t\t\t {true_targets_val_epoch[label]} \t\t\t {correct_predictions_val_epoch[label]}")
            print("Max memory allocated: " + str(torch.cuda.max_memory_allocated()) + "; Memory allocated: " + str(torch.cuda.memory_allocated()))


        # Early
        # early stopping:
        if val_accuracy >= best_val_acc_so_far:
            best_classifier_so_far = copy.deepcopy(classifier)
            best_optimizer_state_so_far = copy.deepcopy(optimizer.state_dict())
            best_classifier_after_num_epochs = number_of_epochs_trained
            best_val_acc_so_far = val_accuracy
            best_classifier_val_loss = val_mean_loss
            best_classifier_training_loss = train_mean_loss
            best_classifier_training_acc = train_accuracy
            epochs_without_improvement_counter = 0
        else:
            epochs_without_improvement_counter += 1

        if epochs_without_improvement_counter != "NEVER":
            if epochs_without_improvement_counter == experiment["EARLY_STOPPING_AFTER"]:
                print(f"Early stopping criterion met. No improvement after {experiment['EARLY_STOPPING_AFTER']} epochs")
                break



        # Train til convergence:
        if (train_accuracy >= 0.8):
            print(f"Model converged. Training stopped.")
            break

except KeyboardInterrupt:
    if PRINTING_FLAG: print("Training canceled!")
    pass



#####################################################################################
# Training finished
#####################################################################################


training_end_time = time.time()
training_time_elapsed = training_end_time - training_start_time


if PRINTING_FLAG:
    print(f"Training finished! Training for {number_of_epochs_trained} epochs took: {training_time_elapsed}s")
    print(f"Training loss: {epochs_train_loss_list}")
    print(f"Validation loss: {epochs_val_loss_list}")
    print(f"Training accuracy: {epochs_train_acc_list}")
    print(f"Validation accuracy: {epochs_val_acc_list}")






#####################################################################################
# Saving results
#####################################################################################

os.mkdir(EXPERIMENT_NAME)

#####################################################################################
# Plotting
#####################################################################################

#epochs_list = [i for i in range(number_of_epochs_trained)]
epochs_trained_list = [i for i in range(len(epochs_train_acc_list))]
epochs_validated_list = [i for i in range(len(epochs_val_acc_list))]


#epochs_list = [i for i in range(number_of_epochs_trained)]
epochs_trained_list = [i for i in range(len(epochs_train_acc_list))]
epochs_validated_list = [i for i in range(len(epochs_val_acc_list))]


plt.plot(epochs_trained_list, epochs_train_acc_list, color="blue", label="Train Accuracy")
plt.plot(epochs_validated_list, epochs_val_acc_list, color="red", label="Validation Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
acc_plot_filename = EXPERIMENT_NAME + "/" + "accuracy_" + EXPERIMENT_NAME + ".png"
plt.savefig(acc_plot_filename)
if PRINTING_FLAG: print(f"Accuracy plot saved at '{acc_plot_filename}'")

plt.clf()

plt.plot(epochs_trained_list, epochs_train_loss_list, color="blue", label="Train Loss")
plt.plot(epochs_validated_list, epochs_val_loss_list, color="red", label="Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
loss_plot_filename = EXPERIMENT_NAME + "/" + "loss_" + EXPERIMENT_NAME + ".png"
plt.savefig(loss_plot_filename)
if PRINTING_FLAG: print(f"Loss plot saved at '{loss_plot_filename}'")

####
#### TODO: confusion matrix
####

#####################################################################################
# Saving checkpoint
#####################################################################################


checkpoint_filename = EXPERIMENT_NAME + "/" + "checkpoint_" + EXPERIMENT_NAME + ".pth"
torch.save({
    'classifier_state_dict': classifier.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_filename)
if PRINTING_FLAG: print(f"Checkpoint saved at '{checkpoint_filename}'")

best_checkpoint_filename = EXPERIMENT_NAME + "/" + "best_" + "checkpoint_" + EXPERIMENT_NAME + ".pth"
torch.save({
    'classifier_state_dict': best_classifier_so_far.state_dict(),
    'optimizer_state_dict': best_optimizer_state_so_far,
    'achieved_after' : best_classifier_after_num_epochs,
    'best_val_acc_so_far' : best_val_acc_so_far,
    'best_classifier_val_loss' : best_classifier_val_loss,
    'best_classifier_training_acc' : best_classifier_training_acc,
    'best_classifier_training_loss' : best_classifier_training_loss,
    }, best_checkpoint_filename)
if PRINTING_FLAG: print(f"Best checkpoint saved at '{best_checkpoint_filename}'")


#####################################################################################
# Output log
#####################################################################################
output_log_string = EXPERIMENT_NAME + "\n"
output_log_string += str(experiment) + "\n\n"
output_log_string += "Number of epochs trained: " + str(number_of_epochs_trained) + f" took {training_time_elapsed}s \n"
output_log_string += f"The best validation result was obtained after {best_classifier_after_num_epochs} epochs \n"
output_log_string += "Checkpoint saved at: " + checkpoint_filename
output_log_string += "Training took: " + str(training_time_elapsed) + "s in total.\n\n"
output_log_string += "Training Loss: \n" + str(epochs_train_loss_list) + "\n\n"
output_log_string += "Validation Loss: \n" + str(epochs_val_loss_list) + "\n\n"
output_log_string += "Training Acc: \n" + str(epochs_train_acc_list) + "\n\n"
output_log_string += "Validation Acc: \n" + str(epochs_val_acc_list) + "\n\n\n\n"
for i in range(len(labels_predicted_train_epochs_list)):
    output_log_string += f"\n Epoch {i}: \n"
    output_log_string += "Training \n"
    output_log_string += f"Labels predicted: \t True targets: \t Correct labels: \n"
    for label in labels_predicted_train_epochs_list[i]:
        output_log_string += f"{labels_predicted_train_epoch[label]} \t\t\t {true_targets_train_epoch[label]} \t\t\t {correct_predictions_train_epoch[label]} \n"
    output_log_string += "\n Validation \n"
    output_log_string += f"Labels predicted: \t True targets: \t Correct labels: \n"
    for label in labels_predicted_val_epochs_list[i]:
        output_log_string += f"{labels_predicted_val_epoch[label]} \t\t\t {true_targets_val_epoch[label]} \t\t\t {correct_predictions_val_epoch[label]} \n"


output_log_filename = EXPERIMENT_NAME + "/" + "output_log_" + EXPERIMENT_NAME + ".txt"
with open(output_log_filename, 'w') as file:
    file.write(output_log_string)
if PRINTING_FLAG: print(f"Output logfile saved at {output_log_filename}")


best_classifier_so_far.eval()
with torch.no_grad():
    test_losses, test_predictions, test_targets = [], [], []

    for batch_number, batch in enumerate(test_dataloader):
        batch.to(device)

        lm_outputs = lm(batch["input_ids"])
        classifier_outputs = best_classifier_so_far(lm_outputs[0].float())

        loss = loss_fn(classifier_outputs, batch["labels"])
        test_losses.append(loss.item())

        test_predictions.extend(classifier_outputs.detach().argmax(dim=1).to('cpu').tolist())
        test_targets.extend(batch["labels"].to('cpu').tolist())

    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    num_predictions_test_epoch = len(test_predictions)
    num_predictions_test_epoch_correct = np.sum(test_predictions == test_targets)
    test_accuracy = num_predictions_test_epoch_correct / num_predictions_test_epoch
    #epochs_test_acc_list.append(test_accuracy)


    test_mean_loss = np.mean(test_losses)

    print(f"{EXPERIMENT_NAME}")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_mean_loss}")

