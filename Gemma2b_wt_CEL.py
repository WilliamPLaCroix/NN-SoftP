from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
import string
import os
from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model, BnbQuantizationConfig
from accelerate import Accelerator
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

os.environ["HF_HOME"] = "/data/users/hhwang/.cache"

dataset = load_dataset("liar")
df = pd.DataFrame(dataset['train'])
df = df.rename(columns={'label': 'label', 'statement': 'statement'})
unique_labels = df['label'].unique()

# For Authentication #
# export HF_HOME=/path/to/cache
# export TRANSFORMERS_CACHE=/path/to/cache
# export HF_DATASETS_CACHE=/path/to/cache
# export HF_METRICS_CACHE=/path/to/cache
# export HF_TOKEN=your_token_here

model_name = 'google/gemma-2b'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
#gemma_model = AutoModelForCausalLM.from_pretrained(model_name)
gemma_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
gemma_tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

max_length_1 = 87

def generate_features(statement, device):
    # Tokenize the statement and ensure the input is on the correct device
    inputs = gemma_tokenizer(statement, return_tensors="pt", max_length=max_length_1, truncation=True, pad_to_max_length=False)
    inputs = inputs.to(device)

    with torch.no_grad():
        # Get model outputs, requesting hidden states
        outputs = gemma_model(**inputs, output_hidden_states=True)

    # Extract embeddings and ensure they are on the correct device
    # This gets the last layer's hidden states, which are used as the token embeddings.
    embeddings = outputs.hidden_states[-1].squeeze(0).to(device)

    return embeddings


# Train Data
df_sampled = df.sample(frac=1, random_state=42)

label_to_int = {label: idx for idx, label in enumerate(df_sampled['label'].unique())}
df_sampled['encoded_labels'] = df_sampled['label'].apply(lambda x: label_to_int[x])
#train_labels = df_sampled['encoded_labels'].tolist()
#train_labels = y_train.tolist()
#val_labels = y_val.tolist()
#test_labels = y_test.tolist()


# Split the data into features and labels
X = df_sampled.drop(columns=['encoded_labels', 'label'])  # Assuming 'label' is the only column to drop
y = df_sampled['encoded_labels']

# Split the data into training (60%) and temp (40%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Split the temp data into validation (50% of temp, 20% of total) and testing (50% of temp, 20% of total) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# X_train, y_train for training
# X_val, y_val for validation
# X_test, y_test for testing

# Generate training features
train_features = [generate_features(statement, device) for statement in tqdm(X_train['statement'], desc="Generating train embeddings")]
# Generate validation features
val_features = [generate_features(statement, device) for statement in tqdm(X_val['statement'], desc="Generating validation embeddings")]
# Generate test features
test_features = [generate_features(statement, device) for statement in tqdm(X_test['statement'], desc="Generating test embeddings")]

# Convert Labels to Tensor
train_labels_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
val_labels_tensor = torch.tensor(y_val, dtype=torch.long, device=device)
test_labels_tensor = torch.tensor(y_test, dtype=torch.long, device=device)

# Find Max Length and Pad Features
max_length = max(max([feature.shape[0] for feature in train_features]),
                 max([feature.shape[0] for feature in val_features]),
                 max([feature.shape[0] for feature in test_features]))

padded_train_features = [torch.nn.functional.pad(feature, (0, 0, 0, max_length - feature.size(0))) for feature in train_features]
padded_val_features = [torch.nn.functional.pad(feature, (0, 0, 0, max_length - feature.size(0))) for feature in val_features]
padded_test_features = [torch.nn.functional.pad(feature, (0, 0, 0, max_length - feature.size(0))) for feature in test_features]

padded_train_features_tensor = torch.stack(padded_train_features)
padded_val_features_tensor = torch.stack(padded_val_features)
padded_test_features_tensor = torch.stack(padded_test_features)

# Create DataLoader for Train/Val/Test
train_dataset = TensorDataset(padded_train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(padded_val_features_tensor, val_labels_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(padded_test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2048, out_channels=128, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(num_features=128)  # Batch normalization after convolution
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=5, stride=5)
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer

        # Placeholder for the flattened size (to be updated dynamically if needed)
        self.flattened_size = None  # This will be set dynamically based on input

        self.fc1 = None  # Placeholder for the first fully connected layer, will be initialized later
        self.bn2 = None  # Placeholder for batch normalization, will be initialized later
        self.fc2 = nn.Linear(64, num_classes)  # Initialization of fc2, assuming fc1's output size is 64

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        # Dynamically calculate flattened size if not set
        if self.fc1 is None:
            self.flattened_size = x.shape[1] * x.shape[2]  # Compute the flattened size
            self.fc1 = nn.Linear(self.flattened_size, 64).to(x.device)  # Initialize fc1 dynamically on the correct device
            self.bn2 = nn.BatchNorm1d(num_features=64).to(x.device)  # Initialize batch normalization for fc1

        x = x.view(-1, self.flattened_size)  # Flatten the output for the fully connected layer
        x = self.dropout(F.relu(self.bn2(self.fc1(x))))  # Apply BN, ReLU, and dropout sequentially
        x = self.fc2(x)
        return x
    
    model = CNN(num_classes=len(label_to_int)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.transpose(1, 2))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(correct / total)

    # Validation loop with progress bar
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # Wrap the val_loader with tqdm for a progress bar
        for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.transpose(1, 2))
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_losses.append(total_loss / len(val_loader))
    val_accuracies.append(correct / total)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")


# Evaluate the model on the test data
model.eval()
test_loss = 0.0
correct_preds = 0
total_preds = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.transpose(1, 2))
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

test_loss /= len(test_loader)
test_acc = correct_preds / total_preds

print(f"Test loss: {test_loss:.3f}.. Test accuracy: {test_acc:.3f}")

plt.figure(figsize=(12, 4))

epochs_range = range(1, epochs + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()