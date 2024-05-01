import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import softmax

# Load the audio tensor
with open('C:/Users/eleni/Data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)

# Load the visual tensor
with open('C:/Users/eleni/Data/visual_features2.pkl', 'rb') as f:
    visual_features = pickle.load(f)

# Load the labels
labels = np.load('C:/Users/eleni/Data/labels.npy')
tlabels = torch.from_numpy(labels)

multimodal = torch.cat((audio_features, visual_features), dim=1)

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

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(-1, 282, 2)  # Reshape to [batch_size, segments, classes]
        # x = softmax(x, dim=-1)
        return x  # Return logits for calibration and softmax application externally
    
# Instantiate the model
model = DepressionPredictor()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

def train_model(model, features, labels, optimizer, criterion, epochs=10):
    model.train()
    probabilities_list = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)  # Outputs are now probabilities for two classes for each segment
        loss = criterion(outputs.view(-1, 2), labels.view(-1))  # Reshape appropriately if needed
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        probabilities_list.append(outputs.detach())
    print(probabilities_list)
    return probabilities_list

labels = torch.full((1, 282), tlabels[0], dtype=torch.long)  # label should be 0 or 1, fill labels with 282 spots of 1 or 0
# Calibrate the model
temperature_model = TemperatureScaler(model)
train_model(temperature_model.model, multimodal, labels, optimizer, nn.CrossEntropyLoss())

# # Calibration
# calibrate_temperature_scaler(temperature_model, validation_features, validation_labels)

# # Inference
# new_data_features = ...  # Load or prepare new data
# probabilities = predict_with_model(temperature_model, new_data_features)