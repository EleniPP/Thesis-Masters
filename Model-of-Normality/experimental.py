import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import json
from pprint import pprint
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import zarr
import torch.optim.lr_scheduler as lr_scheduler
import csv
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from temperature_scaling import ModelWithTemperature

# Load the audio tensor
audio_features = np.load('D:/Data/audio_features1.npy', allow_pickle=True)

# Load the visual tensor
visual_features = np.load('D:/Data/visual_features.npy', allow_pickle=True)
# Load the labels
# labels = np.load('V:/staff-umbrella/EleniSalient/Data/labels.npy')
with open('D:/Data/labels.json', 'r') as file:
    labels_dict = json.load(file)
del labels_dict['492']

# tlabels = torch.from_numpy(labels)
# temp_labels = torch.stack([tlabels[0], tlabels[10]])
# print(temp_labels)
# numbers = list(labels_dict.keys())
numbers = [int(key) for key in labels_dict.keys()]
print(numbers[99])
print(numbers[117])

labels = list(labels_dict.values())
tlabels = torch.tensor(labels)
# Convert audio and visual features to torch tensors
audio_features = [torch.from_numpy(a) for a in audio_features]
visual_features = [torch.from_numpy(v) for v in visual_features]
# multimodal = torch.cat((audio_features, visual_features), dim=1)
# Concatenate audio and visual features for each entry
multimodal_features = [torch.cat((a, v), dim=1) for a, v in zip(audio_features, visual_features)]

# Normalize each tensor individually
normalized_multimodal = []
for tensor in multimodal_features:
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    normalized_tensor = (tensor - mean) / (std + 1e-5)
    normalized_multimodal.append(normalized_tensor)

# Identify and remove NaN entries and flatten inputs for trainig
cleaned_multimodal_features = []
cleaned_labels = []
for i, (features, label) in enumerate(zip(normalized_multimodal, labels)):
    if not torch.isnan(features).any() and not torch.isinf(features).any():
        cleaned_multimodal_features.append(features)
        cleaned_labels.append(label)
    else:
        print(f"Removed entry with NaN/Inf values at index {i}")

# Re-assign the cleaned data
normalized_multimodal = cleaned_multimodal_features
tlabels = torch.tensor(cleaned_labels)


def flatten(multimodal_features, labels):
    flattened_features = []
    flattened_labels = []
    for i, (features, label) in enumerate(zip(normalized_multimodal, labels)):
        for segment in features:
            flattened_features.append(segment)
            flattened_labels.append(label)
    # Convert flattened lists to tensors
    flattened_features = torch.stack(flattened_features)
    flattened_labels = torch.tensor(flattened_labels)
    return flattened_features,flattened_labels

def get_split(split):
    base_path = "D:/Data/"
    extension = "_split.csv"
    file = f"{base_path}{split}{extension}"
    patients_list=[]
    max_row = 47
    # with open('C:/Users/eleni/Data/train_split.csv', newline='') as csvfile:
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        row_count = 0
        for row in spamreader:
            if (row_count >= max_row) and (split == "test"):
                break
            if row[0] == '402' or row[0] == '420':
                row_count += 1
                continue
            patients_list.append(row[0])
            row_count += 1
    patients_array = np.array(patients_list).astype(np.int32)
    return patients_array

def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for features, labels in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
                        
            # labels_flat = labels.view(-1)
            # outputs_flat = outputs.view(-1, 2)
            # valid_mask = labels_flat != -100
            # labels_flat = labels_flat[valid_mask]
            # outputs_flat = outputs_flat[valid_mask]
            loss = criterion(outputs, labels)
            # loss = criterion(outputs.view(-1, 2), labels.view(-1))
            test_loss += loss.item()
            
            # Store predictions and labels for further evaluation (e.g., accuracy, precision, recall)
            all_predictions.append(outputs)
            all_labels.append(labels)
    
            # Calculate accuracy for this batch
            # Convert logits to probabilities and predictions
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            _, predicted = torch.max(outputs, -1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    # Combine predictions and labels from all batches
    all_predictions = np.array(all_predictions,dtype=object)
    all_labels = np.array(all_labels,dtype=object)
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    # precision = precision_score(all_labels, all_predictions)
    # recall = recall_score(all_labels, all_predictions)
    # f1 = f1_score(all_labels, all_predictions)
    # auc = roc_auc_score(all_labels, all_predictions)
    return test_loss / len(dataloader),accuracy, all_predictions, all_labels

def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for features, labels in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
            loss = criterion(outputs.view(-1, 2), labels.view(-1))
            val_loss += loss.item()
    return val_loss / len(dataloader)

# Function to filter features and labels based on patient numbers
def filter_by_patient_numbers(features, labels, patient_numbers):
    indices = [patient_to_index[patient] for patient in patient_numbers if patient in patient_to_index]
    filtered_features = [features[idx] for idx in indices]
    filtered_labels = [labels[idx] for idx in indices]
    return filtered_features, filtered_labels

class DepressionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Function to collate and pad sequences within a batch
def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignore index in loss
    return data_padded, labels_padded

class DepressionPredictor1(nn.Module):
    def __init__(self):
        super(DepressionPredictor1, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(3072, 2048),
            nn.Linear(3072, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),  # Assuming some dropout for regularization
            # nn.Linear(2048, 512),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.6),  # Dropout to prevent overfitting
            # nn.Linear(512, 2) #2 for softmax and 1 for sigmoid
            nn.Linear(256, 2)
        )
# possibleTODO - batch size change?
    def forward(self, x):
        x = self.classifier(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found nan or inf in model output")
        # _, num_segments, _ = x.shape
        # x = x.view(-1, num_segments, 2)  # Reshape to [batch_size, segments, classes]
        return x  # Return logits for calibration and softmax application externally
    

def entropy(probabilities):
    return -(probabilities * torch.log(probabilities)).sum(dim=-1)


def plot_saliency_map(saliency_values, title='Saliency Map'):
    plt.figure(figsize=(10, 5))
    plt.plot(saliency_values, label='Saliency')
    plt.xlabel('Segment')
    plt.ylabel('Normalized Saliency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_losses(train_losses,val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

def normalize_saliency(saliency_values):
    max_val = torch.max(saliency_values)
    min_val = torch.min(saliency_values)
    normalized_saliency = (saliency_values - min_val) / (max_val - min_val)
    return normalized_saliency

def train_model(model, dataloader, optimizer, criterion,epochs=10):
    model.train()
    all_probabilities = []
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        total_loss = 0
        epoch_probabilities = []
        for i, (features,labels) in enumerate(dataloader):
            optimizer.zero_grad()
            # DEBUG: Ensure features do not contain NaN or Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            outputs = model(features)  # Outputs are now probabilities for two classes for each segment
# ----------------------------Previous version -------------------------------------
            # # Flatten labels and outputs to apply loss
            # labels_flat = labels.view(-1)
            # outputs_flat = outputs.view(-1, 2)

            # # Filter out the ignore_index labels and corresponding outputs
            # valid_mask = labels_flat != -100
            # labels_flat = labels_flat[valid_mask]
            # outputs_flat = outputs_flat[valid_mask]
# -----------------------------------------------------------------------------------
            probabilities = softmax(outputs, dim=-1) #i NEED TO CHECK WHERE THE SOFTMAX GOES HERE OR IN THE TEMPERATURE SCALER
            # loss = criterion(outputs.view(-1, 2), labels[i].view(-1))  # Reshape appropriately if needed
            loss = criterion(outputs,labels)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # retain_graph=True
            optimizer.step()
            total_loss += loss.item()    
            # store probabilities for all the patients
            epoch_probabilities.append(probabilities.detach())
        # store probabilities across all epochs
        all_probabilities.append(epoch_probabilities) 
        val_loss = validate_model(model, val_loader, criterion)
        # print(f'Epoch {epoch+1}, Loss: {total_loss}')
        train_losses.append(total_loss/len(dataloader))
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(dataloader)}, Validation Loss: {val_loss}')
        # Step the scheduler
        # scheduler.step(val_loss)
    plot_losses(train_losses,val_losses)
    # all_probabilities = np.array(all_probabilities, dtype=object)
    # np.save('V:/staff-umbrella/EleniSalient/Data/probability_distributions.npy', all_probabilities)
    return all_probabilities

# Use the temp labels since we only have 2 patients

# labels = [torch.full((features.shape[0],), tlabels[i], dtype=torch.long) for i, features in enumerate(normalized_multimodal)]

# Instantiate the model
model = DepressionPredictor1()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
# 5e-3
# Define the learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# Calibrate the model

# Get splits
train_split = get_split('train')
test_split = get_split('test')
val_split = get_split('dev')
del numbers[117]
del numbers[99]
# Create a dictionary mapping patient numbers to their corresponding index
patient_to_index = {patient_num: idx for idx, patient_num in enumerate(numbers)}
for i, features in enumerate(normalized_multimodal):
    if torch.isnan(features).any():
        print(f"NaN values in normalized_multimodal before dataloaders at index {i}")

# Filter the features and labels for each split
train_multimodal, train_labels = filter_by_patient_numbers(normalized_multimodal, labels, train_split)
train_multimodal,train_labels = flatten(train_multimodal,train_labels)
print(train_multimodal.shape)
print(train_multimodal[0].shape)

val_multimodal, val_labels = filter_by_patient_numbers(normalized_multimodal, labels, val_split)
val_multimodal,val_labels = flatten(val_multimodal,val_labels)
print(val_multimodal.shape)
print(val_multimodal[0].shape)
# Leave the test alone for now :)
test_multimodal, test_labels = filter_by_patient_numbers(normalized_multimodal, labels, test_split)
test_multimodal,test_labels = flatten(test_multimodal,test_labels)
print(test_multimodal.shape)
print(test_multimodal[0].shape)
# Create training, validation, and test datasets and dataloaders
train_dataset = DepressionDataset(train_multimodal, train_labels)
val_dataset = DepressionDataset(val_multimodal, val_labels)
test_dataset = DepressionDataset(test_multimodal, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# Do i need to shuffle the validation set as well?
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# Leave the test alone
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss(ignore_index=-100)

probability_distribution = train_model(model, train_loader, optimizer, criterion)

# # Evaluate the model on the test set
# test_loss,accuracy, test_predictions, test_labels = evaluate_model(model, test_loader, criterion)
# print(f'Test Loss: {test_loss}')
# print(f'Accuracy: {accuracy}')
# print(f'Test Precision: {test_precision:.4f}')
# print(f'Test Recall: {test_recall:.4f}')
# print(f'Test F1 Score: {test_f1:.4f}')
# print(f'Test AUC: {test_auc:.4f}')

# Try calibration model from github
scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(val_loader)