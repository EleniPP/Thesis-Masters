import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# log_mels = np.load('V:/staff-umbrella/EleniSalient/Preprocessing/log_mels.npy', allow_pickle=True)

visuals = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/aus.npy', allow_pickle=True)
print('Action Units')
print(visuals.shape)
print(len(visuals[0]))
print(len(visuals[1]))
is_numeric = np.vectorize(lambda x: isinstance(x, (int, float)))(visuals)
if not is_numeric.all():
    print("Non-numeric data detected in visuals array.")
# print(visuals.shape)
non_numeric_indices = np.where(~is_numeric)
for idx in non_numeric_indices[0]:  # Only show the first 10 for inspection
    print(f"Non-numeric element at index {idx}")

class AU1DCNN(nn.Module):
    def __init__(self):
        super(AU1DCNN, self).__init__()
        # Adjust the in_channels to 20 (for the 20 AUs)
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=256, kernel_size=7, padding=3)
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
        # Input x shape: [num_of_patients, num_segments_per_patient, 105, 20]
        # We need to swap dimensions to make it compatible with Conv1d:
        # New shape for Conv1d: [num_of_patients * num_segments_per_patient, 20 (in_channels), 105 (sequence_length)]
        batch_size, num_segments, num_frames, num_aus = x.shape
        
        # Reshape: combine patients and segments, swap AUs and frames
        x = x.view(-1, num_aus, num_frames)
        
        # Apply Conv1D layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten the output
        x = self.flatten(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Input shape [num_of_patients, num_segments_per_patient, 105, 20]
model = AU1DCNN()
model.eval()  # Set the model to evaluation mode

# Convert your 'visuals' array (which contains all patients' AU data) to a tensor
visuals = np.array(visuals)  # Ensure visuals is in the correct shape
# Check if all elements are numeric
visuals = torch.from_numpy(visuals).float()  # Convert to PyTorch tensor

# Assuming the visuals have shape [num_patients, num_segments, 105, 20]
num_patients, num_segments, num_frames, num_aus = visuals.shape

# Reshape for the 1D-CNN: [batch_size (patients * segments), 20 (AUs), 105 (frames)]
visuals = visuals.view(num_patients * num_segments, num_frames, num_aus).permute(0, 2, 1)  # Shape: [num_patients * num_segments, 20, 105]

# Perform feature extraction
with torch.no_grad():
    features = model(visuals)

# Reshape the features back to [num_patients, num_segments, features] if necessary
extracted_features = features.view(num_patients, num_segments, -1)

extracted_features = np.array(extracted_features)
print('Extracted features shape: {}'.format(extracted_features.shape))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/extracted_visual_features.npy', extracted_features)