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
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

# Load the audio tensor
audio_features = np.load('D:/Data/audio_features.npy', allow_pickle=True)
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
labels = list(labels_dict.values())
tlabels = torch.tensor(labels)

# Convert audio and visual features to torch tensors
audio_features = [torch.from_numpy(a) for a in audio_features]
visual_features = [torch.from_numpy(v) for v in visual_features]
# multimodal = torch.cat((audio_features, visual_features), dim=1)
# Concatenate audio and visual features for each entry
multimodal = [torch.cat((a, v), dim=1) for a, v in zip(audio_features, visual_features)]


multimodal_features = [features for i, features in enumerate(multimodal, start=300) if i in numbers]
print(len(multimodal_features))
print(len(multimodal_features[1]))
print(len(multimodal_features[0][0]))

# Normalize each tensor individually
normalized_multimodal = []
for tensor in multimodal_features:
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    normalized_tensor = (tensor - mean) / (std + 1e-5)
    normalized_multimodal.append(normalized_tensor)

class DepressionPredictor(nn.Module):
    def __init__(self):
        super(DepressionPredictor, self).__init__()
        # Input features are the concatenated features of size 2564 (4 audio + 2560 visual)
        self.classifier = nn.Sequential(
            nn.Linear(2564, 2048),
            # nn.Linear(2564, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Assuming some dropout for regularization
            nn.Linear(2048, 512),
            # nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to prevent overfitting
            nn.Linear(512, 2) #2 for softmax and 1 for sigmoid
            # nn.Linear(256, 2)
        )
# possibleTODO - batch size change?
    def forward(self, x):
        x = self.classifier(x)
        _, num_segments, _ = x.shape
        x = x.view(-1, num_segments, 2)  # Reshape to [batch_size, segments, classes]
        # x = softmax(x, dim=-1)
        return x  # Return logits for calibration and softmax application externally
    

class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super(TemperatureScaler, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        scaled_logits = logits / self.temperature  # Scale logits by temperature Zi/T instead of Zi
        probabilities = torch.softmax(scaled_logits, dim=-1)  # Apply softmax along the last dimension
        return probabilities

    def calibrate(self, logits, labels):
        # Define the calibration criterion: Negative Log Likelihood
        # TODO it never goes here so maybe calibration not working properly
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

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


def normalize_saliency(saliency_values):
    max_val = torch.max(saliency_values)
    min_val = torch.min(saliency_values)
    normalized_saliency = (saliency_values - min_val) / (max_val - min_val)
    return normalized_saliency

def train_model(model, multimodal_features, labels, optimizer, criterion, scheduler, epochs=20):
    model.train()
    all_probabilities = []
    for epoch in range(epochs):
        total_loss = 0
        epoch_probabilities = []
        for i, features in enumerate(multimodal_features):
            optimizer.zero_grad()
            # DEBUG: Ensure features do not contain NaN or Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                # print(f'Invalid feature values detected at index {i} ')
                # : {features}
                continue
            features = features.unsqueeze(0)  # Add batch dimension / not sure yet
            outputs = model(features)  # Outputs are now probabilities for two classes for each segment
            probabilities = softmax(outputs, dim=-1) #i NEED TO CHECK WHERE THE SOFTMAX GOES HERE OR IN THE TEMPERATURE SCALER
            loss = criterion(outputs.view(-1, 2), labels[i].view(-1))  # Reshape appropriately if needed
            loss.backward()
            # retain_graph=True
            optimizer.step()
            total_loss += loss.item()    
            # store probabilities for all the patients
            epoch_probabilities.append(probabilities.detach())
        # store probabilities across all epochs
        all_probabilities.append(epoch_probabilities) 
        print(f'Epoch {epoch+1}, Loss: {total_loss}')
        # Step the scheduler
        scheduler.step(total_loss)
    all_probabilities = np.array(all_probabilities, dtype=object)
    np.save('V:/staff-umbrella/EleniSalient/Data/probability_distributions.npy', all_probabilities)
    return all_probabilities

# Use the temp labels since we only have 2 patients

labels = [torch.full((features.shape[0],), tlabels[i], dtype=torch.long) for i, features in enumerate(multimodal_features)]

# Instantiate the model
model = DepressionPredictor()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# Calibrate the model
temperature_model = TemperatureScaler(model)
probability_distribution = train_model(temperature_model.model, normalized_multimodal, labels, optimizer, nn.CrossEntropyLoss(),scheduler)





