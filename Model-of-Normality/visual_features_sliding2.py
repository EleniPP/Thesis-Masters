import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

# visuals = np.load('V:/staff-umbrella/EleniSalient/Data/aggr_visual.npy', allow_pickle=True)
# print(visuals.shape) #(2, 282, 24)
# Define the directory where the files re saved
visual_dir = '../../../tudelft.net/staff-umbrella/EleniSalient/aggr_visual_sliding/'
# List all the files in the directory
visual_files = [f for f in os.listdir(visual_dir) if f.startswith('visual_') and f.endswith('.npy')]

# Sort the files if necessary (depends on how they are named, e.g., log_mel_1, log_mel_2, etc.)
visual_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Initialize an empty list to store the arrays
visual_segments = []

# Load and append each array to the list
for file in visual_files:
    file_path = os.path.join(visual_dir, file)
    visual_segments.append(np.load(file_path))

# Combine all arrays into one
visuals = np.concatenate(visual_segments, axis=0)

print(f"Combined log-mel shape: {visuals.shape}")
# print(type(input_data[0][0]))
class AU1DCNN(nn.Module):
    def __init__(self, num_features):
        super(AU1DCNN, self).__init__()
        # First set of layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.05)
        
        # Second set of layers
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.05)
        
        self.flatten = nn.Flatten()
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # If you intend to use it for classification or other purposes, you might flatten the output here
        # and pass it to a fully connected layer (not included in your specifications).
        x = self.flatten(x)
        
        # Returning x directly for now as it contains the features after the last dropout layer
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
# Example usage:
model = AU1DCNN(num_features=1)
print(model)

# Train the model (example: assuming binary classification)

all_features = []
for visual in visuals:
    filtered_visual = visual[:, 4:].astype(np.float32)
    tvisual = torch.from_numpy(filtered_visual)

    input_data = tvisual.unsqueeze(1)  # Add channel dimension
    with torch.no_grad():
        model.eval()
        features = model(input_data)
        all_features.append(features.numpy())

# Convert list of numpy arrays to a single numpy array with dtype=object
all_features_np = np.array(all_features, dtype=object)

np.save('../../../tudelft.net/staff-umbrella/EleniSalient/visual_features_sliding_2.npy', all_features_np)
# # Save a tensor
# with open('C:/Users/eleni/Data/visual_features2.pkl', 'wb') as f:
#     pickle.dump(features, f)

# print(features)