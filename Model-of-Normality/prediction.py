import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the audio tensor
with open('C:/Users/eleni/Data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)

# Load the visual tensor
with open('C:/Users/eleni/Data/visual_features2.pkl', 'rb') as f:
    visual_features = pickle.load(f)

# Load the labels
labels = np.load('C:/Users/eleni/Data/labels.npy')
tlabels = torch.from_numpy(labels)

print(tlabels.shape)

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
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.classifier(x)
        return torch.sigmoid(x)  # Apply sigmoid activation function to output layer

# Instantiate the model
model = DepressionPredictor()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)