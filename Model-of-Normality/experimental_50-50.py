import torch
import pickle
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import csv
from temperature_scaling import ModelWithTemperature
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from torch.optim.lr_scheduler import StepLR
import h5py
from sklearn.utils import resample

def load_features_from_hdf5(hdf5_file_path):
    patient_features = []

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        for patient_id in hdf5_file:
            # Load each patient's features into a NumPy array
            patient_data = hdf5_file[patient_id][:]
            print(f"Loaded features for {patient_id} with shape {patient_data.shape}")
            patient_features.append(patient_data)

    # Convert the list of features to a NumPy array with dtype=object
    feature_patients = np.array(patient_features, dtype=object)
    return feature_patients

# Example usage to load the features
# audio_features = load_features_from_hdf5('D:/Data/log_mel_sliding/test_audio_features2_sliding.h5')
audio_features = load_features_from_hdf5('../../../tudelft.net/staff-umbrella/EleniSalient/log_mel_sliding/audio_features2_sliding.h5')
# audio_features = load_features_from_hdf5('../../../tudelft.net/staff-umbrella/EleniSalient/audio_features2_sliding.h5')
print(f"Loaded audio features for all patients. Shape: {audio_features.shape}")

# visual_features = load_features_from_hdf5('D:/Data/aggr_visual_sliding/test_visual_features_sliding2.h5')
visual_features = load_features_from_hdf5('../../../tudelft.net/staff-umbrella/EleniSalient/aggr_visual_sliding/visual_features_sliding2.h5')
print(f"Loaded visual features for all patients. Shape: {visual_features.shape}")

print(audio_features.shape)
print(audio_features[0].shape)

print(visual_features.shape)
print(visual_features[0].shape)

with open('../../../tudelft.net/staff-umbrella/EleniSalient/Data/labels.json', 'r') as file:
    labels_dict = json.load(file)
# Only for the test
del labels_dict['492']
# Print or work with the first three labels
print(labels_dict)

numbers = [int(key) for key in labels_dict.keys()]


labels = list(labels_dict.values())
tlabels = torch.tensor(labels)
# Convert audio and visual features to torch tensors
audio_features = [torch.from_numpy(a) for a in audio_features]
visual_features = [torch.from_numpy(v) for v in visual_features]

multimodal_features = [torch.cat((a, v), dim=1) for a, v in zip(audio_features, visual_features)]
# print(multimodal_features.shape)
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
    num_seg_per_patient = []
    segment_indices = []  # To track original chronological order within each patient
    for i, (features, label) in enumerate(zip(multimodal_features, labels)):
        cnt = 0
        for idx,segment in enumerate(features):
            cnt += 1
            flattened_features.append(segment)
            flattened_labels.append(label)
            segment_indices.append(idx)
        num_seg_per_patient.append(cnt)
    # Convert flattened lists to tensors
    flattened_features = torch.stack(flattened_features)
    flattened_labels = torch.tensor(flattened_labels)
    segment_per_patient = torch.tensor(num_seg_per_patient)
    segment_indices = torch.tensor(segment_indices) #so this contains the index of each segment for all the segments in the train multimodal
    return flattened_features,flattened_labels,segment_per_patient, segment_indices

def get_split(split):
    base_path = "../../../tudelft.net/staff-umbrella/EleniSalient/"
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

def evaluate_model_cross(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    threshold = 0.5
    with torch.no_grad():  # Disable gradient calculation
        for features, labels in dataloader:
        # for the final train evaluation / because I have different data loaders.
        # for features, labels,_,_ in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
            loss = criterion(outputs, labels)
            # loss = xon(outputs.view(-1, 2), labels.view(-1))
            test_loss += loss.item()

            # Calculate accuracy for this batch
            # Convert logits to probabilities and predictions
            probs = torch.softmax(outputs, dim=1)[:,1]  # Probability of class 1
            # _, predicted = torch.max(outputs, -1)
            predicted = (probs > threshold).int()
            # Append predictions and labels for later evaluation
            all_predictions.append(predicted)  # Move to CPU if using GPU
            all_labels.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Concatenate all predictions and labels from all batches
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    print(classification_report(all_labels, all_predictions))
    return test_loss / len(dataloader),accuracy, all_predictions, all_labels

def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    threshold = 0.5
    with torch.no_grad():  # Disable gradient calculation
        for features, labels,_,_ in dataloader:
        # for the final train evaluation / because I have different data loaders.
        # for features, labels,_,_ in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
            loss = criterion(outputs, labels)
            # loss = xon(outputs.view(-1, 2), labels.view(-1))
            test_loss += loss.item()

            # Calculate accuracy for this batch
            # Convert logits to probabilities and predictions
            probs = torch.softmax(outputs, dim=1)[:,1]  # Probability of class 1
            # _, predicted = torch.max(outputs, -1)
            predicted = (probs > threshold).int()
            # Append predictions and labels for later evaluation
            all_predictions.append(predicted)  # Move to CPU if using GPU
            all_labels.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Concatenate all predictions and labels from all batches
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    print(classification_report(all_labels, all_predictions))
    return test_loss / len(dataloader),accuracy, all_predictions, all_labels

def validate_model_cross(model, dataloader, criterion):
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

def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for features, labels,_,_ in dataloader:
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights, should be a tensor with shape [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # No reduction to compute per-sample loss

        # Get the predicted probabilities for the correct class (pt)
        pt = torch.exp(-ce_loss)  # Probabilities of the true class

        # If alpha (class weights) are provided, apply them
        if self.alpha is not None:
            # Ensure alpha is properly broadcasted (shape [batch_size])
            at = self.alpha.gather(0, targets)  # Gather correct alpha for each target in the batch
            ce_loss = at * ce_loss  # Apply class weights to the cross-entropy loss

        # Apply focal loss scaling
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply the reduction (mean, sum, etc.)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # Return raw losses if no reduction is needed

class DepressionDatasetCross(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DepressionDataset(Dataset):
    def __init__(self, data, labels, patient_numbers, segment_indices):
        self.data = data
        self.labels = labels
        self.patient_numbers = patient_numbers
        self.segment_indices = segment_indices  # Original segment order

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.patient_numbers[idx], self.segment_indices[idx]

# Function to collate and pad sequences within a batch
def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignore index in loss
    return data_padded, labels_padded

class DepressionPredictor1(nn.Module):
    def __init__(self):
        super(DepressionPredictor1, self).__init__()
        # Completely simplified:
        self.classifier = nn.Sequential(
            nn.Linear(5632, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        # self.classifier = nn.Sequential(
        #     # nn.Linear(5632, 1024),
        #     nn.Linear(5632, 2048),   # Increase the number of units in the first layer
        #     nn.ReLU(),
        #     nn.Dropout(0.5),  # Slightly reduce dropout to avoid too much regularization
        #     nn.Linear(2048, 1024),   # New additional layer
        #     # nn.Linear(3072, 512),
        #     # nn.BatchNorm1d(1024),  # Batch normalization
        #     nn.ReLU(),
        #     nn.Dropout(0.5),  # Assuming some dropout for regularization
        #     nn.Linear(1024, 512),   # Continue down to smaller layers
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),    
        #     # nn.Linear(512, 128),
        #     # nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),  # Dropout to prevent overfitting
        #     nn.Linear(256, 2) #2 for softmax and 1 for sigmoid
        #     # nn.Linear(128, 2)
        # )
# possibleTODO - batch size change?
    def forward(self, x):
        x = self.classifier(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found nan or inf in model output")
        # _, num_segments, _ = x.shape
        # x = x.view(-1, num_segments, 2)  # Reshape to [batch_size, segments, classes]
        return x  # Return logits for calibration and softmax application externally


def plot_losses(train_losses,val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/loss_plot.png')
    plt.close()


def train_final_model(model, dataloader,optimizer,scheduler, criterion, epochs = 30):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for i, (features, labels,_,_) in enumerate(dataloader):
            optimizer.zero_grad()
            # DEBUG: Ensure features do not contain NaN or Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            outputs = model(features)  # Outputs are logits for two classes
            loss = criterion(outputs, labels)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
        # Logging the average training loss for this epoch
        avg_train_loss = total_loss / len(dataloader)
        scheduler.step(avg_train_loss)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')

    # Save the final trained model
    # for the test comment
    torch.save(model.state_dict(), '../../../tudelft.net/staff-umbrella/EleniSalient/final_model1.pth')
    print("Final model saved as 'final_model1.pth'")

    # Optionally, plot the training losses
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/training_loss_plot.png')
    plt.close()

    return model

def train_model(model, dataloader,val_loader, optimizer, scheduler,criterion,epochs=30):
    # Early stopping
    early_stopping_patience = 10  # Number of epochs with no improvement after which training will be stopped
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (features,labels,_,_) in enumerate(dataloader):
            optimizer.zero_grad()
            # DEBUG: Ensure features do not contain NaN or Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            outputs = model(features)  # Outputs are now probabilities for two classes for each segment
            # loss = criterion(outputs.view(-1, 2), labels[i].view(-1))  # Reshape appropriately if needed
            loss = criterion(outputs,labels)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # retain_graph=True
            optimizer.step()
            total_loss += loss.item()
            # store probabilities for all the patients

        # TODO: me poio val kano edo validate? poios val_loader einai aftos?
        val_loss = validate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model if it's the best one yet
            # for the test comment
            torch.save(model.state_dict(), '../../../tudelft.net/staff-umbrella/EleniSalient/best_model1.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
        # print(f'Epoch {epoch+1}, Loss: {total_loss}')
        train_losses.append(total_loss/len(dataloader))
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(dataloader)}, Validation Loss: {val_loss}')
        # Step the scheduler
        # scheduler.step(val_loss)
    plot_losses(train_losses,val_losses)
    # all_probabilities = np.array(all_probabilities, dtype=object)
    # np.save('V:/staff-umbrella/EleniSalient/Data/probability_distributions.npy', all_probabilities)
    return model

def train_model_cross(model, dataloader,val_loader, optimizer, scheduler,criterion,epochs=30):
    # Early stopping
    early_stopping_patience = 10  # Number of epochs with no improvement after which training will be stopped
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (features,labels) in enumerate(dataloader):
            optimizer.zero_grad()
            # DEBUG: Ensure features do not contain NaN or Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            outputs = model(features)  # Outputs are now probabilities for two classes for each segment
            # loss = criterion(outputs.view(-1, 2), labels[i].view(-1))  # Reshape appropriately if needed
            loss = criterion(outputs,labels)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # retain_graph=True
            optimizer.step()
            total_loss += loss.item()
            # store probabilities for all the patients

        # TODO: me poio val kano edo validate? poios val_loader einai aftos?
        val_loss = validate_model_cross(model, val_loader, criterion)
        scheduler.step(val_loss)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model if it's the best one yet
            # for the test comment
            torch.save(model.state_dict(), '../../../tudelft.net/staff-umbrella/EleniSalient/best_model1.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
        # print(f'Epoch {epoch+1}, Loss: {total_loss}')
        train_losses.append(total_loss/len(dataloader))
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(dataloader)}, Validation Loss: {val_loss}')
        # Step the scheduler
        # scheduler.step(val_loss)
    plot_losses(train_losses,val_losses)
    # all_probabilities = np.array(all_probabilities, dtype=object)
    # np.save('V:/staff-umbrella/EleniSalient/Data/probability_distributions.npy', all_probabilities)
    return model
# Use the temp labels since we only have 2 patients

# labels = [torch.full((features.shape[0],), tlabels[i], dtype=torch.long) for i, features in enumerate(normalized_multimodal)]

# Instantiate the model
model = DepressionPredictor1()

# # Define the optimizer
# # optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=2e-4)
# # optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# # 5e-3
# # Define the learning rate scheduler
# # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3, verbose=True)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Calibrate the model

# Get splits
# for test comment
train_split = get_split('train')
test_split = get_split('test')
val_split = get_split('dev')
# # for the test
train_split = train_split[:-1] #because i didnt put the 491 in the labels_dict and in the data

# train_split = np.array([303,304])
# val_split = np.array([302])
# test_split = np.array([300])
# Only for test
del numbers[117]
del numbers[99]
# Create a dictionary mapping patient numbers to their corresponding index

patient_to_index = {patient_num: idx for idx, patient_num in enumerate(numbers)}
for i, features in enumerate(normalized_multimodal):
    if torch.isnan(features).any():
        print(f"NaN values in normalized_multimodal before dataloaders at index {i}")

# Filter the features and labels for each split


train_multimodal, train_labels = filter_by_patient_numbers(normalized_multimodal, labels, train_split)
print(type(train_multimodal))
train_multimodal,train_labels,segments_per_patient_train,segments_order_train = flatten(train_multimodal,train_labels)
#Check and fix inbalance
unique, counts = np.unique(train_labels, return_counts=True)
class_distribution = dict(zip(unique, counts))

print(f"Class distribution: {class_distribution}")

# Determine majority and minority class
majority_class = max(class_distribution, key=class_distribution.get)
minority_class = min(class_distribution, key=class_distribution.get)

# Get indices of the majority and minority classes
majority_indices = np.where(train_labels == majority_class)[0]
minority_indices = np.where(train_labels == minority_class)[0]

n_majority = len(majority_indices)
n_minority = len(minority_indices)

# Undersample the majority class to match the number of minority class samples
majority_undersampled_indices = resample(majority_indices, 
                                         replace=False,  # No replacement, undersample without duplication
                                         n_samples=n_minority,  # Match the number of minority class samples
                                         random_state=42)

# Combine the minority and undersampled majority indices
balanced_indices = np.concatenate([majority_undersampled_indices, minority_indices])

# Shuffle the indices to ensure randomization
np.random.shuffle(balanced_indices)

balanced_indices = torch.tensor(balanced_indices, dtype=torch.long)  # Convert indices to tensor if not already

# Index train_multimodal and train_labels using balanced_indices
train_multimodal = train_multimodal[balanced_indices]
train_labels = train_labels[balanced_indices]

# balanced_indices = list(balanced_indices)  # If balanced_indices is a NumPy array, convert it to a list
# # Filter train_multimodal and train_labels using list comprehension
# train_multimodal = [train_multimodal[i] for i in balanced_indices]
# train_labels = [train_labels[i] for i in balanced_indices]


# Check the new class distribution
unique_balanced, counts_balanced = np.unique(train_labels, return_counts=True)
class_distribution_balanced = dict(zip(unique_balanced, counts_balanced))
print(f"Balanced class distribution: {class_distribution_balanced}")
# array that has the patient number for each segment in the train multimnodal
segments_patients_train = [num for count, num in zip(segments_per_patient_train, train_split) for _ in range(count)]


val_multimodal, val_labels = filter_by_patient_numbers(normalized_multimodal, labels, val_split)
val_multimodal,val_labels,segments_per_patient_val,segments_order_val = flatten(val_multimodal,val_labels)
segments_patients_val = [num for count, num in zip(segments_per_patient_val, val_split) for _ in range(count)]
# print(val_multimodal.shape)
# print(val_multimodal[0].shape)
# Leave the test alone for now :)
test_multimodal, test_labels = filter_by_patient_numbers(normalized_multimodal, labels, test_split)
test_multimodal,test_labels,segments_per_patient_test,segments_order_test = flatten(test_multimodal,test_labels)
segments_patients_test = [num for count, num in zip(segments_per_patient_test, test_split) for _ in range(count)]
# print(test_multimodal.shape)
# print(test_multimodal[0].shape)
# Create training, validation, and test datasets and dataloaders
train_dataset = DepressionDataset(train_multimodal, train_labels,segments_patients_train,segments_order_train)
val_dataset = DepressionDataset(val_multimodal, val_labels,segments_patients_val,segments_order_val)
test_dataset = DepressionDataset(test_multimodal, test_labels,segments_patients_test,segments_order_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# Do i need to shuffle the validation set as well?
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
# Leave the test alone
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# criterion = nn.CrossEntropyLoss(ignore_index=-100)

# probability_distribution = train_model(model, train_loader,val_loader, optimizer, criterion)


# -----------CROSS VALIDATION-------------------------------------
# # Number of folds for cross-validation
# n_splits = 5 #for the debugging of the temperature scaling

# # Initialize KFold
# kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Assuming train_multimodal and train_labels are already in the shape (number_of_segments x features)
# # and labels respectively
# features = train_multimodal  # Use your multimodal features directly
# labels = train_labels  # Use your labels directly
# # Store results for each fold
# fold_results = []

# # Cross-Validation Loop
# for fold, (train_index, val_index) in enumerate(kf.split(features)):
#     print(f"Fold {fold + 1}/{n_splits}")

#     # Split data into training and validation based on indices
#     train_features, val_features = features[train_index], features[val_index]
#     train_labels, val_labels = labels[train_index], labels[val_index]


#     # Assuming y contains your labels (for the entire dataset)
#     unique, counts = np.unique(train_labels, return_counts=True)
#     class_distribution = dict(zip(unique, counts))
#     print(class_distribution)

#     # Initialize SMOTE
#     # smote = SMOTE(sampling_strategy=0.5)
#     # Assuming you have your training data in X_train and y_train
#     # adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=2)
#     # Try to solve the imbalance problem with oversampling
#     # train_features, train_labels = smote.fit_resample(train_features, train_labels)

#     # Create DataLoaders for this fold
#     train_dataset = DepressionDatasetCross(train_features, train_labels)
#     val_dataset = DepressionDatasetCross(val_features, val_labels)

#     training_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#     valid_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

#     # Initialize a new model for this fold
#     model = DepressionPredictor1()

#     # Define the optimizer and learning rate scheduler
#     # optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

#     optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Adam default learning rate is 0.001
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
#     # Initialize the scheduler (e.g., StepLR)
#     # scheduler = StepLR(optimizer, step_size=8, gamma=0.7)

#     # Define the loss function
#     # criterion = nn.CrossEntropyLoss(ignore_index=-100,weight=torch.tensor([1.0, 1.2]))
#     # alpha = torch.tensor([1.0, 1.2])  # Weights for class 0 and class
#     # when it goes up like 1.3 from 1.2 then precision is higher for class 1 but recall is lower
#     # criterion = FocalLoss(alpha=alpha, gamma=1.6)
#     criterion= nn.CrossEntropyLoss()
#     # Train the model for this fold
#     probability_distribution = train_model_cross(model, training_loader,valid_loader, optimizer,scheduler, criterion)

#     # Evaluate on the validation set
#     val_loss, accuracy, _, _ = evaluate_model_cross(model, valid_loader, criterion)

#     # Store results for this fold
#     fold_results.append({
#         'fold': fold + 1,
#         'val_loss': val_loss,
#         'accuracy': accuracy
#     })

#     print(f"Validation Loss for fold {fold + 1}: {val_loss:.4f}")
#     print(f"Validation Accuracy for fold {fold + 1}: {accuracy:.4f}")

# # Calculate average validation loss and accuracy across all folds
# avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
# avg_accuracy = np.mean([result['accuracy'] for result in fold_results])

# print(f"Average Validation Loss across {n_splits} folds: {avg_val_loss:.4f}")
# print(f"Average Validation Accuracy across {n_splits} folds: {avg_accuracy:.4f}")


# Final training of the model in the whole training set
# Assuming you have your full training data in train_loader
final_model = DepressionPredictor1()  # Initialize your model architecture
# Initialize weights using Xavier initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
        if m.bias is not None:
            m.bias.data.fill_(0.01)

final_model.apply(init_weights)

# Define optimizer and criterion
optimizer = optim.Adam(final_model.parameters(), lr=1e-4, weight_decay=1e-4)
alpha = torch.tensor([1.0, 1.2])
criterion = FocalLoss(alpha=alpha, gamma=1.0)
# class_weights = torch.tensor([2.0,1.0])
# criterion = nn.CrossEntropyLoss()

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
# Train the model on the full dataset
# final_model = train_final_model(final_model, train_loader, optimizer, scheduler, criterion, epochs=30)
# final_model = train_model(model, train_loader,val_loader, optimizer,scheduler, criterion)
final_model = train_model(model, train_loader,val_loader, optimizer,scheduler, criterion)

# Evaluate on the validation set
val_loss, accuracy, _, _ = evaluate_model(model, val_loader, criterion)
print(1/0)
# Load the saved model
model = DepressionPredictor1()  # Initialize your model architecture
model.load_state_dict(torch.load('../../../tudelft.net/staff-umbrella/EleniSalient/final_model1.pth'))
model.eval()  # Set the model to evaluation mode
print("Model loaded and ready for calibration")
# Try calibration model from github
scaled_model = ModelWithTemperature(model)
scaled_model.set_temperature(val_loader)

scaled_model.eval()
all_probs = []
all_patient_numbers = []
all_segment_orders = []
with torch.no_grad():
    for inputs, _ , patient_numbers,segments_order in train_loader:
        logits = scaled_model(inputs)  # Scaled logits
        probs = torch.softmax(logits, dim=1)  # Calibrated probabilities
        all_probs.append(probs)
        # Save patient numbers from the current batch (same order as the probabilities predicted for the same segments)
        all_patient_numbers.extend(patient_numbers.tolist())
        all_segment_orders.extend(segments_order.tolist())
# Combine probabilities from all batches
all_probs = torch.cat(all_probs, dim=0)
# Convert patient numbers list to a tensor
all_patient_numbers = torch.tensor(all_patient_numbers)
all_segment_orders = torch.tensor(all_segment_orders)

# for the test
torch.save(all_probs, '../../../tudelft.net/staff-umbrella/EleniSalient/probability_distributions.pth')
torch.save(all_patient_numbers, '../../../tudelft.net/staff-umbrella/EleniSalient/all_patient_numbers.pth')
torch.save(all_segment_orders, '../../../tudelft.net/staff-umbrella/EleniSalient/all_segment_orders.pth')