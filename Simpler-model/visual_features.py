import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


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
        # # Input x shape: [num_of_patients, num_segments_per_patient, 105, 20]
        # # We need to swap dimensions to make it compatible with Conv1d:
        # # New shape for Conv1d: [num_of_patients * num_segments_per_patient, 20 (in_channels), 105 (sequence_length)]
        # batch_size, num_segments, num_frames, num_aus = x.shape
        
        # # Reshape: combine patients and segments, swap AUs and frames
        # x = x.view(-1, num_aus, num_frames)
        
        # Input x shape for each patient: [num_segments, 105, 20]
    
        # Permute to match Conv1D input shape: [num_segments, 20 (AUs), 105 (frames)]
        # x = x.permute(0, 2, 1)

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

# Dimensionality reduction model
class FeatureReducer(nn.Module):
    def __init__(self, input_dim, reduced_dim):
        super(FeatureReducer, self).__init__()
        self.fc = nn.Linear(input_dim, reduced_dim)

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    # log_mels = np.load('V:/staff-umbrella/EleniSalient/Preprocessing/log_mels.npy', allow_pickle=True)

    visuals = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/aus_reliable_prep.npy', allow_pickle=True)
    print('Action Units')
    print(visuals.shape)
    print(len(visuals[0]))
    print(len(visuals[1]))
    # I STILL HAVEN'T ADDED THE MLP AT THE END FOR DIMENTIONALITY REDUCTION SOS
    # Input shape [num_of_patients, num_segments_per_patient, 105, 20]
    model = AU1DCNN()
    model.eval()  # Set the model to evaluation mode

    input_dim = 13312  # Size of extracted features
    output_dim = 512   # Reduced dimensionality (adjust based on your needs)
    reducer = FeatureReducer(input_dim, output_dim)
    reducer.eval()  # Set to evaluation mode for now

    # List to hold extracted features for each patient
    all_extracted_features = []
    all_reduced_features = []
    # Iterate over each patient
    for patient_idx, patient_data in enumerate(visuals):
        try:
            # Ensure patient data is numeric and convert to float32
            patient_data = np.array(patient_data, dtype=np.float32)
            
            # Check if the patient data has the correct shape
            if patient_data.shape[1:] == (105, 20):
                # Reshape for 1D-CNN and permute to [num_segments, 20, 105]
                patient_tensor = torch.from_numpy(patient_data).view(-1, 105, 20).permute(0, 2, 1)  # Shape: [num_segments, 20, 105]

                # Extract features with the model
                with torch.no_grad():
                    features = model(patient_tensor)
                    reduced_features = reducer(features)  # Shape: [num_segments, output_dim]

                print('Features')
                print(features.shape)
                print(features[0].shape)
                print(type(features[0]))


                # Apply dimensionality reduction
                print('Reduced Features')
                print(reduced_features.shape)
                print(reduced_features[0].shape)  
                print(type(reduced_features[0]))

                # Collect features for this patient
                all_extracted_features.append(features.numpy())
                # Collect reduced features for this patient
                all_reduced_features.append(reduced_features.numpy())
            else:
                print(f"Skipping patient {patient_idx} due to incorrect shape: {patient_data.shape}")
        except Exception as e:
            print(f"Error processing patient {patient_idx}: {e}")



    extracted_features = np.array(all_extracted_features, dtype=object)
    reduced_features = np.array(all_reduced_features, dtype=object)

    for idx, patient_features in enumerate(all_reduced_features):
        if len(patient_features) == 0 or not all(isinstance(segment, np.ndarray) for segment in patient_features):
            print(f"Problematic patient {idx}: {patient_features}")

    # print('Extracted Features')
    # print(extracted_features.shape)
    # print(all_extracted_features[0].shape)
    # print(all_extracted_features[0][0].shape)
    # print(all_extracted_features[0][0][0].shape)

    # print('Reduced Features')
    # print(reduced_features_array.shape)
    # print(all_reduced_features[0].shape)
    # print(all_reduced_features[0][0].shape)
    # print(all_reduced_features[0][0][0].shape)

    # print('Extracted Features')
    # print(extracted_features.shape)
    # print(len(extracted_features[0]))
    # print(all_extracted_features[0][0][0].shape)

    # print('Reduced Features')
    # print(reduced_features_array.shape)
    # print(len(reduced_features_array[0]))
    # print(all_reduced_features[0][0][0].shape)
    # I STILL HAVEN'T ADDED THE MLP AT THE END FOR DIMENTIONALITY REDUCTION SOS
    # np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/extracted_visual_features.npy', extracted_features)
    np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/extracted_visual_features_reduced_reliable_prep.npy', reduced_features)
