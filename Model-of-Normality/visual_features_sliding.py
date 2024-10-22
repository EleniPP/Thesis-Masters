import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import h5py
import os

# visuals = np.load('V:/staff-umbrella/EleniSalient/Data/aggr_visual.npy', allow_pickle=True)
# print(visuals.shape) #(2, 282, 24)
# Save each patient's data into HDF5
def save_to_hdf5(numbers, base_path="D:/Data/aggr_visual_sliding", data_type="visual", output_file="test_combined_visual_all_patients.h5"):
    with h5py.File(f"{base_path}/{output_file}", 'w') as hdf5_file:
        patient_group = hdf5_file.create_group("patients")
        
        for number in numbers:
            data_file_path = f"{base_path}/{data_type}_{number}.npy"
            if os.path.exists(data_file_path):
                print(f"Loading {data_type} data for patient {number}...")
                patient_data = np.load(data_file_path, allow_pickle=True)
                patient_group.create_dataset(f"patient_{number}", data=patient_data)

# Function to save the features incrementally to HDF5 without compression
def save_features_to_hdf5(patient_features, patient_ids, output_file):
    with h5py.File(output_file, 'w') as hdf5_file:
        for patient_id, features in zip(patient_ids, patient_features):
            print(f"Saving features for patient {patient_id} with shape {features.shape}")
            # Save the features using the actual patient ID
            hdf5_file.create_dataset(f"patient_{patient_id}", data=features)

def serialize_array(array):
    return pickle.dumps(array)

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


# if os.path.exists("D:/Data/aggr_visual_sliding/combined_visual_all_patients.h5"):
#     print(f"{'D:/Data/aggr_visual_sliding/combined_visual_all_patients.h5'} already exists. Skipping save_to_hdf5.")
# else:
#     print(f"{'D:/Data/aggr_visual_sliding/combined_visual_all_patients.h5'} does not exist. Saving data...")
#     numbers = list(range(300, 491))  # Patient IDs
#     save_to_hdf5(numbers)  # Save if the file doesn't exist  
# 
# numbers = list(range(300, 303))  # Patient IDs
# for test
numbers = [303,304,302,300]
save_to_hdf5(numbers)
#               
# Example usage:
model = AU1DCNN(num_features=1)
print(model)

# Train the model (example: assuming binary classification)


patient_features = []
with h5py.File('D:/Data/aggr_visual_sliding/test_combined_visual_all_patients.h5', 'r') as hdf5_file:
    patient_group = hdf5_file["patients"]

    sorted_patient_ids = sorted(patient_group.keys(), key=lambda x: int(x.split('_')[1]))
    for patient_id in sorted_patient_ids:
        print(patient_id)
        visual = patient_group[patient_id]  # Load the patient's log-mel data from HDF5
        filtered_visual = visual[:, 4:].astype(np.float32)
        tvisual = torch.from_numpy(filtered_visual)

        input_data = tvisual.unsqueeze(1)  # Add channel dimension
        with torch.no_grad():
            model.eval()
            features = model(input_data)
            patient_features.append(features.numpy())  # Append the NumPy array as it is
            # serialized_features = serialize_array(features.numpy())
            # patient_features.append(Row(patient_id=float(patient_id.split('_')[1]), features=serialized_features))
            # patient_features.append(Row(patient_id=float(patient_id.split('_')[1]), features=features.numpy().flatten().tolist()))  # Flatten the array and convert to list
            # patient_features.append(features.numpy())




save_features_to_hdf5(patient_features, sorted_patient_ids, 'D:/Data/aggr_visual_sliding/test_visual_features_sliding2.h5')
# all_features = []
# for visual in visuals:
#     filtered_visual = visual[:, 4:].astype(np.float32)
#     tvisual = torch.from_numpy(filtered_visual)

#     input_data = tvisual.unsqueeze(1)  # Add channel dimension
#     with torch.no_grad():
#         model.eval()
#         features = model(input_data)
#         all_features.append(features.numpy())

# # Convert list of numpy arrays to a single numpy array with dtype=object
# all_features_np = np.array(all_features, dtype=object)

# np.save('V:/staff-umbrella/EleniSalient/Data/visual_features.npy', all_features_np)
