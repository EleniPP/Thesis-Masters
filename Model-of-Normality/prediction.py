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

multimodal = torch.cat((audio_features, visual_features), dim=1)

class ProbabilisticSegmentPredictor(nn.Module):
    def __init__(self):
        super(ProbabilisticSegmentPredictor, self).__init__()
        # Input features are the concatenated features of size 2564 (4 audio + 2560 visual)
        self.features = nn.Sequential(
            nn.Linear(2564, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),  # Assuming some dropout for regularization
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to prevent overfitting
        )
        self.mean = nn.Linear(512, 1)  # Output mean score for each segment
        self.log_variance = nn.Linear(512, 1)  # Output log variance for each segment

    def forward(self, x):
        # x shape: [282, 2564] - features for each segment
        num_segments, num_features = x.shape
        # batch_size, num_segments, num_features = x.shape // batch_size = 1
        x = x.view(-1, num_features)  # Flatten segments into the batch dimension
        x = self.features(x)
        mean = self.mean(x)
        log_variance = self.log_variance(x)
        mean = mean.view(1, num_segments)  # Reshape to get means per patient per segment
        log_variance = log_variance.view(1, num_segments)  # Reshape for variances per patient per segment
        
        # Calculate probability for each segment using the sigmoid function for the mean
        probability = torch.sigmoid(mean)

        return probability, log_variance
# Instantiate the model
model = ProbabilisticSegmentPredictor()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def custom_loss(probabilities, log_variance, labels):
    # Assume binary cross-entropy for probability with log variance as an uncertainty measure
    # This is a placeholder for a more sophisticated uncertainty-aware loss
    bce_loss = nn.BCELoss(reduction='none')
    variance = torch.exp(log_variance)  # Convert log variance to variance
    loss = bce_loss(probabilities, labels)  # Compute BCE loss per segment
    weighted_loss = torch.mean(loss / variance) + torch.mean(log_variance)  # Weight loss by inverse variance
    return weighted_loss

# Define the binary cross entropy loss
# criterion = nn.BCELoss(reduction='none')  # 'none' keeps the loss per item (segment)
def train_model(model, features, labels, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        probabilities, log_variance = model(features)
        loss = criterion(probabilities, log_variance, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

print(multimodal.shape[0])
labels = torch.full((1, 282), tlabels[0])
# Train the model
train_model(model, multimodal, labels, optimizer, custom_loss)