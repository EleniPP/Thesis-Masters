import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import softmax
import matplotlib.pyplot as plt


# Load the audio tensor
# with open('C:/Users/eleni/Data/audio_features.pkl', 'rb') as f:
#     audio_features = pickle.load(f)
audio_features = np.load('C:/Users/eleni/Data/audio_features.npy', allow_pickle=True)
# Load the visual tensor
# with open('C:/Users/eleni/Data/visual_features2.pkl', 'rb') as f:
#     visual_features = pickle.load(f)
visual_features = np.load('C:/Users/eleni/Data/visual_features.npy', allow_pickle=True)

# Load the labels
labels = np.load('C:/Users/eleni/Data/labels.npy')
tlabels = torch.from_numpy(labels)

# Convert audio and visual features to torch tensors
audio_features = [torch.from_numpy(a) for a in audio_features]
visual_features = [torch.from_numpy(v) for v in visual_features]

# multimodal = torch.cat((audio_features, visual_features), dim=1)
# Concatenate audio and visual features for each entry
multimodal_features = [torch.cat((a, v), dim=1) for a, v in zip(audio_features, visual_features)]

class DepressionPredictor(nn.Module):
    def __init__(self):
        super(DepressionPredictor, self).__init__()
        # Input features are the concatenated features of size 2564 (4 audio + 2560 visual)
        self.classifier = nn.Sequential(
            nn.Linear(2564, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),  # Assuming some dropout for regularization
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to prevent overfitting
            nn.Linear(512, 2) #2 for softmax and 1 for sigmoid
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
        print('Helloooooooo')
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

# ReWRITEEEEEEEEEEEEEEEEEEEEE
def compute_jacobian(inputs, model):
    outputs = model(inputs)  # Get logits from the model
    probabilities = torch.softmax(outputs, dim=-1)  # Apply softmax to get probabilities
    entropies = entropy(probabilities)  # Compute entropy for each segment

    jacobian_matrix = []
    for i in range(entropies.size(1)):  # Loop over each segment
        grad_output = torch.zeros_like(entropies)
        grad_output[:, i] = 1
        # inputs.grad = None  # Clear existing gradients``
        entropies.backward(grad_output, retain_graph=True)  # Compute gradients
        jacobian_matrix.append(inputs.grad.detach().clone())
        inputs.grad.zero_()  # Reset gradients after each step

    jacobian_tensor = torch.stack(jacobian_matrix, dim=1)  # Stack to form the Jacobian matrix
    return jacobian_tensor  # Stack to form the Jacobian matrix


def saliency_from_jacobian(jacobian):
    # Calculate the product of the transpose of the Jacobian and the Jacobian
    jtj = torch.bmm(jacobian.transpose(2, 1), jacobian)
    print("J^T * J values:", jtj)
    # Calculate the determinant of the resulting matrix
    saliency = torch.det(jtj)
    return saliency

def plot_saliency_map(saliency_values, title='Saliency Map'):
    plt.figure(figsize=(10, 5))
    plt.plot(saliency_values.cpu().numpy(), label='Saliency')
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

def train_model(model, features, labels, optimizer, criterion, epochs=1):
    model.train()
    # probabilities_list = []
    # entropy_history = []
    # jacobian_history = []    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)  # Outputs are now probabilities for two classes for each segment
        probabilities = softmax(outputs, dim=-1) #i NEED TO CHECK WHERE THE SOFTMAX GOES HERE OR IN THE TEMPERATURE SCALER
        loss = criterion(outputs.view(-1, 2), labels.view(-1))  # Reshape appropriately if needed
        loss.backward(retain_graph=True)
        optimizer.step()

        entopies = entropy(probabilities)
        print(entopies)
        # Compute Jacobian of entropy with respect to inputs
        jacobian_matrix = compute_jacobian(features.requires_grad_(), model)
      
        print(jacobian_matrix)

        # Calculate saliency from Jacobian
        saliency = saliency_from_jacobian(jacobian_matrix)
        normalized_saliency = normalize_saliency(saliency)

        # Debug prints to verify dimensions and values
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        print(f'Logits shape: {outputs.shape}')
        print(f'Probabilities shape: {probabilities.shape}')
        print(f'Entropy shape: {entropy(probabilities).shape}')
        print(f'Jacobian matrix shape: {jacobian_matrix.shape}')
        print(f'Saliency shape: {saliency.shape}')
        print(f'Normalized Saliency shape: {normalized_saliency.shape}')
        print('NORMALIZED SALIENCY')
        print( saliency)

        # Plot saliency map for each epoch (optional)
        plot_saliency_map(normalized_saliency, title=f'Saliency Map - Epoch {epoch+1}')

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # return probabilities_list
    
labels = torch.full((1, 282), tlabels[0], dtype=torch.long)  # label should be 0 or 1, fill labels with 282 spots of 1 or 0

# Instantiate the model
model = DepressionPredictor()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Calibrate the model
temperature_model = TemperatureScaler(model)
train_model(temperature_model.model, multimodal, labels, optimizer, nn.CrossEntropyLoss())


# # So we need to compute saliency and stuff with the predictions and not during the epochs


# # # Analyze entropy changes for saliency
# # for epoch_entropy in entropy_history:
# #     delta_entropy = epoch_entropy[:, 1:] - epoch_entropy[:, :-1]  # Compute changes in entropy
# #     significant_changes = (delta_entropy.abs() > 0.1).nonzero(as_tuple=True)
# #     print("Significant entropy changes found at segments:", significant_changes[1])


# # # Calibration
# # calibrate_temperature_scaler(temperature_model, validation_features, validation_labels)

# # # Inference
# # new_data_features = ...  # Load or prepare new data
# # probabilities = predict_with_model(temperature_model, new_data_features) 

