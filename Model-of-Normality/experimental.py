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

import zarr


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


# visual = np.load('D:/Data/aggr_visual.npy', allow_pickle=True)

# log_mel = np.load('D:/Data/log_mel.npy', allow_pickle=True)


# audio = [log_mel_i.reshape(log_mel_i.shape[0], -1) for log_mel_i in log_mel]
# # Convert the list of numpy arrays to a list of torch tensors
# # audio = [np.array(a) for a in audio]    

# # audio = [torch.from_numpy(a) for a in audio]
# audio = [torch.tensor(a, dtype=torch.float32) for a in audio]

# # visual = [torch.from_numpy(v) for v in visual]
# visual = [torch.tensor(v, dtype=torch.float32) for v in visual]
# with open('D:/Data/labels.json', 'r') as file:
#     labels_dict = json.load(file)
# del labels_dict['492']

# numbers = [int(key) for key in labels_dict.keys()]
# labels = list(labels_dict.values())
# tlabels = torch.tensor(labels)

# multimodal = [torch.cat((a, v), dim=1) for a, v in zip(audio, visual)]
# multimodal_features = [features for i, features in enumerate(multimodal, start=300) if i in numbers]

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
        self.lstm = nn.LSTM(input_size=2564, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x shape: [batch_size, num_segments, 2564]
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out shape: [batch_size, num_segments, 512 * 2]
        # Aggregate LSTM outputs
        lstm_out = lstm_out[:, -1, :]  # Taking the output of the last segment
        # Pass through the classifier
        out = self.classifier(lstm_out)
        return out  # Logits
    

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

def train_model(model, multimodal_features, labels, optimizer, criterion, epochs=20):
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
    all_probabilities = np.array(all_probabilities, dtype=object)
    np.save('V:/staff-umbrella/EleniSalient/Data/probability_distributions.npy', all_probabilities)
    return all_probabilities

# Use the temp labels since we only have 2 patients

labels = [torch.full((features.shape[0],), tlabels[i], dtype=torch.long) for i, features in enumerate(multimodal_features)]

# Instantiate the model
model = DepressionPredictor()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Calibrate the model
temperature_model = TemperatureScaler(model)
probability_distribution = train_model(temperature_model.model, normalized_multimodal, labels, optimizer, nn.CrossEntropyLoss())








# -----------------------------------------------------------------------------------------------------------
# class MultimodalDataset(Dataset):
#     def __init__(self, multimodal_features, labels):
#         self.multimodal_features = multimodal_features
#         self.labels = labels

#     def __len__(self):
#         return len(self.multimodal_features)

#     def __getitem__(self, idx):
#         return self.multimodal_features[idx], self.labels[idx]



# def normalize(tensor):
#     mean = tensor.mean(dim=0, keepdim=True)
#     std = tensor.std(dim=0, keepdim=True)
#     normalized_tensor = (tensor - mean) / (std + 1e-5)  # Add a small value to avoid division by zero
#     return normalized_tensor


# class DepressionPredictor(nn.Module):
#     def __init__(self):
#         super(DepressionPredictor, self).__init__()
#         # Input features are the concatenated features of size 2564 (4 audio + 2560 visual)
#         self.classifier = nn.Sequential(
#             nn.Linear(2564, 2048),
#             nn.ReLU(),
#             nn.Dropout(0.5),  # Assuming some dropout for regularization
#             nn.Linear(2048, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),  # Dropout to prevent overfitting
#             nn.Linear(512, 2) #2 for softmax and 1 for sigmoid
#         )
# # possibleTODO - batch size change?
#     def forward(self, x):
#         x = self.classifier(x)
#         _, num_segments, _ = x.shape
#         x = x.view(-1, num_segments, 2)  # Reshape to [batch_size, segments, classes]
#         # x = softmax(x, dim=-1)
#         return x  # Return logits for calibration and softmax application externally
    

# class TemperatureScaler(nn.Module):
#     def __init__(self, model):
#         super(TemperatureScaler, self).__init__()
#         self.model = model
#         self.temperature = nn.Parameter(torch.ones(1))

#     def forward(self, x):
#         logits = self.model(x)
#         scaled_logits = logits / self.temperature  # Scale logits by temperature Zi/T instead of Zi
#         probabilities = torch.softmax(scaled_logits, dim=-1)  # Apply softmax along the last dimension
#         return probabilities

#     def calibrate(self, logits, labels):
#         # Define the calibration criterion: Negative Log Likelihood
#         # TODO it never goes here so maybe calibration not working properly
#         # print('Helloooooooo')
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

#         def closure():
#             optimizer.zero_grad()
#             scaled_logits = logits / self.temperature
#             loss = criterion(scaled_logits / self.temperature, labels)
#             loss.backward()
#             return loss

#         optimizer.step(closure)

# def entropy(probabilities):
#     return -(probabilities * torch.log(probabilities)).sum(dim=-1)

# def calculate_entropy(tensor):
#     p = F.softmax(tensor, dim=-1)
#     log_p = F.log_softmax(tensor, dim=-1)
#     entropy = -torch.sum(p * log_p, dim=-1)
#     return entropy

# def plot_saliency_map(saliency_values, title='Saliency Map'):
#     plt.figure(figsize=(10, 5))
#     plt.plot(saliency_values, label='Saliency')
#     plt.xlabel('Segment')
#     plt.ylabel('Normalized Saliency')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# class CustomDataset(Dataset):
#     def __init__(self, data_list, label_list):
#         self.data_list = data_list
#         self.label_list = label_list

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         return self.data_list[idx], self.label_list[idx]
    
# # Custom collate function with normalization and padding using pad_sequence
# def custom_collate_fn(batch):
#     # Separate data and labels
#     data, labels = zip(*batch)
#     # Normalize each tensor in the batch
#     normalized_data = [normalize(tensor) for tensor in data]
#     # Pad the sequences
#     padded_data = pad_sequence(normalized_data, batch_first=True, padding_value=0)
#     # Pad the labels (assuming they are 1D tensors)
#     padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
#     # Create mask
#     lengths = [tensor.size(0) for tensor in data]
#     masks = torch.zeros(padded_data.shape[:2], dtype=torch.bool)
#     for i, length in enumerate(lengths):
#         masks[i, :length] = 1
#     return padded_data, padded_labels, masks, lengths

# def collate_fn(batch):
#     # Unzip the batch into features and labels
#     features, labels = zip(*batch)
    
#     # Compute the lengths of the sequences
#     lengths = [f.size(0) for f in features]
    
#     # Pad the sequences to the maximum length in the batch
#     padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    
#     # Pad the labels
#     padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for padding value for labels (ignored index in loss)
#     # Stack labels
#     # labels = torch.stack(labels)

#     return padded_features, padded_labels, lengths

# def normalize_saliency(saliency_values):
#     max_val = torch.max(saliency_values)
#     min_val = torch.min(saliency_values)
#     normalized_saliency = (saliency_values - min_val) / (max_val - min_val)
#     return normalized_saliency

# class SegmentFeatureExtractor(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(SegmentFeatureExtractor, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, input_size)

#     def forward(self, x, lengths):
#         packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
#         packed_output, (hn, cn) = self.lstm(packed_input)
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
#         output = self.fc(output)
#         return output


# def train_model(model, dataloader, optimizer, criterion, epochs=10):
#     model.train()
#     all_probabilities = []
#     all_logits = []
#     all_labels = []
#     for epoch in range(epochs):
#         total_loss = 0
#         epoch_probabilities = []
#         # for i, features in enumerate(multimodal_features):
#         for batch_idx,batch in enumerate(dataloader):
#             features, labels, lengths = batch
#             optimizer.zero_grad()

#             # print(f"Batch {batch_idx} features before NaN/Inf check:")
#             print(features.size())

#             for i,f in enumerate(features):
#                 print('Feature:')
#                 print(f)
#                 if torch.isnan(f).any() or torch.isinf(f).any():
#                     print(print(f'Invalid feature values detected at index {i} '))
#                     continue

#             # Filter out invalid features
#             # invalid_indices = [i for i, f in enumerate(features) if (torch.isnan(f).any() or torch.isinf(f).any())]
#             # print(invalid_indices)
#             # # Filter out invalid features and labels
#             # valid_features = [f for i, f in enumerate(features) if i not in invalid_indices]
#             # valid_labels = [l for i, l in enumerate(labels) if i not in invalid_indices]
#             # valid_lengths = [l for i, l in enumerate(lengths) if i not in invalid_indices]

#             # # Convert lists back to tensors
#             # if len(valid_features) == 0:
#             #     print(f'No valid features in this batch. Skipping batch.')
#             #     continue
            
#             # valid_features = pad_sequence(valid_features, batch_first=True, padding_value=0)
#             # valid_labels = pad_sequence(valid_labels, batch_first=True, padding_value=-100)
#             # # Pack the padded sequence
#             # packed_features = pack_padded_sequence(valid_features, valid_lengths, batch_first=True, enforce_sorted=False)

#             # packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
#             # features = features.unsqueeze(0)  # Add batch dimension / not sure yet
#             # Forward pass
#             # packed_outputs, (hn, cn) = model(packed_features)
#             outputs = model(features)  # Outputs are now probabilities for two classes for each segment
#             # Unpack the output
#             # unpacked_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
#             probabilities = softmax(outputs, dim=-1) #i NEED TO CHECK WHERE THE SOFTMAX GOES HERE OR IN THE TEMPERATURE SCALER


#             # Compute loss
#             reshaped_outputs = outputs.view(-1, 2)
#             # reshaped_labels = valid_labels.view(-1)
#             reshaped_labels = labels.view(-1)
#             loss = criterion(reshaped_outputs, reshaped_labels)
#             # loss = criterion(outputs.view(-1, 2), labels[i].view(-1))  # Reshape appropriately if needed

#             # loss.backward(retain_graph=True)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()    
#             # store probabilities for all the patients
#             epoch_probabilities.append(probabilities.detach())

#             # Collect logits and labels for calibration
#             all_logits.append(reshaped_outputs.detach())
#             all_labels.append(reshaped_labels.detach())
#         # store probabilities across all epochs
#         all_probabilities.append(epoch_probabilities) 
#         print(f'Epoch {epoch+1}, Loss: {total_loss}')
#     all_probabilities = np.array(all_probabilities, dtype=object)
#     np.save('V:/staff-umbrella/EleniSalient/Data/probability_distributions.npy', all_probabilities)
#     return all_probabilities,torch.cat(all_logits), torch.cat(all_labels)


# def train_model1(model, multimodal_features, labels, optimizer, criterion, epochs=20):
#     model.train()
#     all_probabilities = []
#     for epoch in range(epochs):
#         total_loss = 0
#         epoch_probabilities = []
#         for i, features in enumerate(multimodal_features):
#             optimizer.zero_grad()
#             # DEBUG: Ensure features do not contain NaN or Inf
#             if torch.isnan(features).any() or torch.isinf(features).any():
#                 # print(f'Invalid feature values detected at index {i} ')
#                 # : {features}
#                 continue
#             features = features.unsqueeze(0)  # Add batch dimension / not sure yet
#             outputs = model(features)  # Outputs are now probabilities for two classes for each segment
#             probabilities = softmax(outputs, dim=-1) #i NEED TO CHECK WHERE THE SOFTMAX GOES HERE OR IN THE TEMPERATURE SCALER
#             loss = criterion(outputs.view(-1, 2), labels[i].view(-1))  # Reshape appropriately if needed
#             loss.backward()
#             # retain_graph=True
#             optimizer.step()
#             total_loss += loss.item()    
#             # store probabilities for all the patients
#             epoch_probabilities.append(probabilities.detach())
#         # store probabilities across all epochs
#         all_probabilities.append(epoch_probabilities) 
#         print(f'Epoch {epoch+1}, Loss: {total_loss}')
#         return all_probabilities
# # Use the temp labels since we only have 2 patients

# labels = [torch.full((features.shape[0],), tlabels[i], dtype=torch.long) for i, features in enumerate(multimodal)]
# # labels = torch.full((1, 282), tlabels[0], dtype=torch.long)  # label should be 0 or 1, fill labels with 282 spots of 1 or 0

# dataset = CustomDataset(multimodal, labels)
# data_loader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate_fn)

# # Create dataset and dataloader
# # dataset = MultimodalDataset(multimodal, labels)
# # dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)


# # # Instantiate the model
# model = DepressionPredictor()
# # Initialize the model
# # input_size = 2564
# # hidden_size = 512
# # num_layers = 2
# # model = SegmentFeatureExtractor(input_size, hidden_size, num_layers)


# # Define the optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# # Calibrate the model
# temperature_model = TemperatureScaler(model)
# probability_distribution,logits, labels = train_model1(temperature_model.model, multimodal,labels, optimizer, nn.CrossEntropyLoss())
# # print(len(probability_distribution)) #length is 10 for the epochs
# # Calibrate the model
# temperature_model.calibrate(logits, labels)

# # Print the temperature parameter
# print(f"Calibrated Temperature: {temperature_model.temperature.item()}")



# # Training loop (simplified)
# # for padded_data, padded_labels, masks, lengths in data_loader:
# #     # Forward pass
# #     features = model(padded_data, lengths)
    
# #     # Calculate entropy
# #     entropies = calculate_entropy(features)
    
# #     # Use the mask to ignore padded values
# #     masked_entropies = entropies * masks
    
# #     # Identify salient segments (example thresholding)
# #     threshold = torch.mean(masked_entropies[masks]) + torch.std(masked_entropies[masks])
# #     salient_segments = masked_entropies > threshold
    
# #     # Perform operations with salient segments
# #     print(salient_segments)
# #     # You can use padded_labels for loss calculation here


# # # # comment for the first run / maybe i'll put it in a different file
# # # for probabilities in probability_distribution[9]:
# # #     entropies = entropy(probabilities).numpy().flatten()2
# # #     # print(entropies.shape)
# # #     # Compute gradients
# # #     jacobian = np.gradient(entropies)
# # #     # Since this is a 1d problem, J'(x)J(x) is 1x1, therefore determinant
# # #     # is the square of gradient itself.
# # #     saliency = np.square(jacobian)
# # #     # print(jacobian)
# # #     plot_saliency_map(saliency[0], title=f'Saliency Map')
# # #     # saliency = saliency_from_jacobian(jacobian)
# # #     # print(saliency)




# # # They PROBABLY need after the training
# # # entopies = entropy(probabilities)
# # #         print(entopies)
# # #         # Compute Jacobian of entropy with respect to inputs
# # #         jacobian_matrix = compute_jacobian(features.requires_grad_(), model)
      
# # #         print(jacobian_matrix)

# # #         # Calculate saliency from Jacobian
# # #         saliency = saliency_from_jacobian(jacobian_matrix)
# # #         normalized_saliency = normalize_saliency(saliency)

# # #         # Debug prints to verify dimensions and values
# # #         print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# # #         print(f'Logits shape: {outputs.shape}')
# # #         print(f'Probabilities shape: {probabilities.shape}')
# # #         print(f'Entropy shape: {entropy(probabilities).shape}')
# # #         print(f'Jacobian matrix shape: {jacobian_matrix.shape}')
# # #         print(f'Saliency shape: {saliency.shape}')
# # #         print(f'Normalized Saliency shape: {normalized_saliency.shape}')
# # #         print('NORMALIZED SALIENCY')
# # #         print( saliency)

# # #         # Plot saliency map for each epoch (optional)
# # #         plot_saliency_map(normalized_saliency, title=f'Saliency Map - Epoch {epoch+1}')
# # # ------------------------------------------------------------------------------------------------
# # # So we need to compute saliency and stuff with the predictions and not during the epochs

# # # # Analyze entropy changes for saliency
# # # for epo ch_entropy in entropy_history:
# # #     delta_entropy = epoch_entropy[:, 1:] - epoch_entropy[:, :-1]  # Compute changes in entropy
# # #     significant_changes = (delta_entropy.abs() > 0.1).nonzero(as_tuple=True)
# # #     print("Significant entropy changes found at segments:", significant_changes[1])


# # # # Calibration
# # # calibrate_temperature_scaler(temperature_model, validation_features, validation_labels)

# # # # Inference
# # # new_data_features = ...  # Load or prepare new data
# # # probabilities = predict_with_model(temperature_model, new_data_features) 

