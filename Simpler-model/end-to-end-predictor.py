import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import csv
from temperature_scaling import ModelWithTemperature
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.utils import resample
import torch.optim.lr_scheduler as lr_scheduler

# Import models
from audio_features import ModifiedAlexNet, get_3d_spec, modifiedAlexNet
from visual_features import AU1DCNN, FeatureReducer
from predictor import DepressionPredictor1


# Load the labels
with open('/tudelft.net/staff-umbrella/EleniSalient/labels.json', 'r') as file:
    labels_dict = json.load(file)
del labels_dict['492']

numbers = [int(key) for key in labels_dict.keys()]
labels = list(labels_dict.values())
tlabels = torch.tensor(labels)

# Load the audio and visual
visuals = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/aus_reliable.npy', allow_pickle=True)
log_mel_data = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/log_mels_reliable.npy', allow_pickle=True)

def flatten(multimodal_features, labels):
    flattened_features = []
    flattened_labels = []
    for i, (features, label) in enumerate(zip(multimodal_features, labels)):
        for idx, segment in enumerate(features):
            flattened_features.append(segment)
            flattened_labels.append(label)
    # Convert flattened lists to tensors
    flattened_features = torch.stack(flattened_features)
    flattened_labels = torch.tensor(flattened_labels)
    return flattened_features, flattened_labels

def get_split(split):
    base_path = "/tudelft.net/staff-umbrella/EleniSalient/"
    extension = "_split.csv"
    file = f"{base_path}{split}{extension}"
    patients_list = []
    max_row = 47
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
    threshold = 0.5
    with torch.no_grad():  # Disable gradient calculation
        for features, labels, _, _ in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            predicted = (probs > threshold).int()
            all_predictions.append(predicted)
            all_labels.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = correct / total if total > 0 else 0
    print(classification_report(all_labels, all_predictions))
    return test_loss / len(dataloader), accuracy, all_predictions, all_labels

def validate_model(audio_model, visual_model, reducer, predictor, val_loader, criterion):
    predictor.eval()  # Set the predictor to evaluation mode
    val_loss = 0.0  # Initialize validation loss accumulator

    with torch.no_grad():  # Disable gradient calculation
        for batch_audio, batch_visual, batch_labels in val_loader:
            # Ensure no NaN or Inf in the batch
            if torch.isnan(batch_audio).any() or torch.isinf(batch_audio).any() or \
               torch.isnan(batch_visual).any() or torch.isinf(batch_visual).any():
                continue

            # Feature Extraction for Audio
            log_mel_spec_3d = torch.stack([get_3d_spec(segment.numpy()) for segment in batch_audio])
            log_mel_spec_3d = log_mel_spec_3d.permute(0, 3, 1, 2)  # Shape: [batch_size, 3, 64, 351]
            audio_features = audio_model(log_mel_spec_3d)  # Shape: [batch_size, audio_feature_dim]

            # Feature Extraction for Visual
            batch_visual = batch_visual.permute(0, 2, 1)  # Shape: [batch_size, 20, 105]
            visual_features = visual_model(batch_visual)  # Shape: [batch_size, visual_feature_dim]
            reduced_visual_features = reducer(visual_features)  # Shape: [batch_size, reduced_dim]

            # Combine Features
            combined_features = torch.cat((audio_features, reduced_visual_features), dim=1)  # [batch_size, combined_dim]

            # Normalize and flatten the combined features
            normalized_multimodal = normalize_clean_data(combined_features)

            # Skip if NaN/Inf
            if torch.isnan(normalized_multimodal).any() or torch.isinf(normalized_multimodal).any():
                continue

            # Make Predictions
            outputs = predictor(normalized_multimodal)  # Shape: [batch_size, num_classes]

            # Compute Loss
            loss = criterion(outputs.view(-1, 2), batch_labels.view(-1))  # Match output and label shapes
            val_loss += loss.item()

    # Return average validation loss
    return val_loss / len(val_loader)


def filter_by_patient_numbers(features, labels, patient_numbers):
    indices = [patient_to_index[patient] for patient in patient_numbers if patient in patient_to_index]
    filtered_features = [features[idx] for idx in indices]
    filtered_labels = [labels[idx] for idx in indices]
    return filtered_features, filtered_labels

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            ce_loss = at * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DepressionDatasetCross(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DepressionDataset(Dataset):
    def __init__(self, labels, audio_data, visual_data):
        self.labels = labels
        self.audio_data = audio_data
        self.visual_data = visual_data


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.audio_data[idx],self.visual_data[idx]

def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return data_padded, labels_padded


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()


def normalize_clean_data(multimodal_features):
    # Normalize each tensor individually
    normalized_multimodal = []
    for tensor in multimodal_features:
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        normalized_tensor = (tensor - mean) / (std + 1e-5)
        normalized_multimodal.append(normalized_tensor)

    # Identify and remove NaN entries and flatten inputs for training
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
    return normalized_multimodal



# Get splits
train_split = get_split('train')
test_split = get_split('test')
val_split = get_split('dev')
train_split = train_split[:-1]

del numbers[117]
del numbers[99]
patient_to_index = {patient_num: idx for idx, patient_num in enumerate(numbers)}

#Train Pipeline

def train_pipeline(audio_model, visual_model, reducer, predictor, dataloader, val_loader, optimizer, scheduler, criterion, epochs=30):
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
    audio_model, visual_model, reducer, predictor = (
        audio_model.to(device),
        visual_model.to(device),
        reducer.to(device),
        predictor.to(device),
    )

    for epoch in range(epochs):
        # Set models to training mode
        audio_model.train()
        visual_model.train()
        reducer.train()
        predictor.train()
        total_loss = 0
        for batch_idx, (batch_labels, batch_audio, batch_visual) in enumerate(dataloader):
            optimizer.zero_grad()
                

            # Move data to device
            batch_audio, batch_visual, batch_labels = (
                batch_audio.to(device),
                batch_visual.to(device),
                batch_labels.to(device),
            )

            # Feature Extraction for Audio
            # Flattened audio data is already [batch_size, 64, 351]
            log_mel_spec_3d = torch.stack([get_3d_spec(segment.numpy()) for segment in batch_audio])
            log_mel_spec_3d = log_mel_spec_3d.permute(0, 3, 1, 2)  # Shape: [batch_size, 3, 64, 351]
            audio_features = audio_model(log_mel_spec_3d)  # Shape: [batch_size, audio_feature_dim]

            # Feature Extraction for Visual
            # Flattened visual data is already [batch_size, 105, 20]
            batch_visual = batch_visual.permute(0, 2, 1)  # Shape: [batch_size, 20, 105]
            visual_features = visual_model(batch_visual)  # Shape: [batch_size, visual_feature_dim]
            reduced_visual_features = reducer(visual_features)  # Shape: [batch_size, reduced_dim]

            # Combine Features
            combined_features = torch.cat((audio_features, reduced_visual_features), dim=1)  # Shape: [num_segments, combined_feature_dim]

            # Normalize and flatten the combined features
            normalized_multimodal = normalize_clean_data(combined_features)

            # Skip if NaN/Inf
            if torch.isnan(normalized_multimodal).any() or torch.isinf(normalized_multimodal).any():
                continue

            # Predictor Training
            outputs = predictor(normalized_multimodal)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        val_loss = validate_model(audio_model, visual_model, reducer, predictor, val_loader, criterion)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(predictor.state_dict(), "/tudelft.net/staff-umbrella/EleniSalient/best_model.pth")
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
        train_losses.append(total_loss / len(dataloader))
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss / len(dataloader)}, Validation Loss: {val_loss}')
    plot_losses(train_losses, val_losses)
    return model



# Split the data
# Audio is [num_patients,num_segments,64,351]
# Visual is [num_patients,num_segments,105,20]

train_visuals, train_labels = filter_by_patient_numbers(visuals, labels, train_split)
train_visuals, flatlabels = flatten(train_visuals, train_labels)
print(flatlabels.shape)
train_audio, train_labels = filter_by_patient_numbers(log_mel_data, labels, train_split)
train_audio, train_labels = flatten(train_audio, train_labels)
print(train_labels.shape)

val_visuals, val_labels = filter_by_patient_numbers(visuals, labels, val_split)
val_visuals, _ = flatten(val_visuals, val_labels)
val_audio, val_labels = filter_by_patient_numbers(log_mel_data, labels, val_split)
val_audio, val_labels = flatten(val_audio, val_labels)

test_visuals, test_labels = filter_by_patient_numbers(visuals, labels, test_split)
test_visuals, _ = flatten(test_visuals, test_labels)
test_audio, test_labels = filter_by_patient_numbers(log_mel_data, labels, test_split)
test_audio, test_labels = flatten(test_audio, test_labels)


train_dataset = DepressionDataset(train_labels,train_audio,train_visuals)
val_dataset = DepressionDataset(val_labels,val_audio,val_visuals)
test_dataset = DepressionDataset(test_labels,test_audio,test_visuals)

train_loader = DataLoader(train_dataset, batch_size=128,shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = DepressionPredictor1()
audio_model = modifiedAlexNet(pretrained=False)
visual_model = AU1DCNN()

input_dim = 13312  # Size of extracted features
output_dim = 512   # Reduced dimensionality (adjust based on your needs)
reducer = FeatureReducer(input_dim, output_dim)

# why no SGD?
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
# alpha = torch.tensor([1.0, 1.3])
# criterion = FocalLoss(alpha=alpha, gamma=1.6)
criterion = nn.CrossEntropyLoss()

model = train_pipeline(audio_model, visual_model, reducer, model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=30) 
# model = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion)

# Evaluate on the validation set
val_loss, accuracy, _, _ = evaluate_model(model, val_loader, criterion)