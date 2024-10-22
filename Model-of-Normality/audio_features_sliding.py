import numpy as np
import torch 
import torchvision.models as models
import tensorflow as tf
import tensorflow_hub as hub
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import h5py
import os

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# log_mel_seg = np.load('D:/Data/log_mel.npy', allow_pickle=True)
# print(log_mel_seg.shape)
# print(log_mel_seg[0].shape)


N_CHANNELS = 3

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.num_classes=num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((12, 12))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        print('features',x.shape)
        
        #x = self.avgpool(x)
        #print('avgpool',x.shape)
        #x = torch.flatten(x, 1)
        #print('flatten',x.shape)
        #x = self.classifier(x)
        return x

def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],progress=progress)
        model.load_state_dict(state_dict)
    return model
     

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=3072):
        super(ModifiedAlexNet, self).__init__()
        self.num_classes=num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        #print('features',x.shape)
        x=torch.flatten(x, start_dim=2)#a1,a2,a3......al{a of dim c} 
        x=torch.sum(x, dim=2)#a1*alpha1+a2*alpha2+.......+al*alphal
        x=self.classifier(x)
        return x
   
def modifiedAlexNet(pretrained=False, progress=True, **kwargs):
    model_modified = ModifiedAlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],progress=progress)
        model_modified.load_state_dict(state_dict)
    return model_modified



def get_3d_spec(Sxx_in, moments=None):
    if moments is not None:
        (base_mean, base_std, delta_mean, delta_std,
             delta2_mean, delta2_std) = moments
    else:
        base_mean, delta_mean, delta2_mean = (0, 0, 0)
        base_std, delta_std, delta2_std = (1, 1, 1)
    h, w = Sxx_in.shape
    right1 = np.concatenate([Sxx_in[:, 0].reshape((h, -1)), Sxx_in], axis=1)[:, :-1]
    delta = (Sxx_in - right1)[:, 1:]
    delta_pad = delta[:, 0].reshape((h, -1))
    delta = np.concatenate([delta_pad, delta], axis=1)
    right2 = np.concatenate([delta[:, 0].reshape((h, -1)), delta], axis=1)[:, :-1]
    delta2 = (delta - right2)[:, 1:]
    delta2_pad = delta2[:, 0].reshape((h, -1))
    delta2 = np.concatenate([delta2_pad, delta2], axis=1)
    base = (Sxx_in - base_mean) / base_std
    delta = (delta - delta_mean) / delta_std
    delta2 = (delta2 - delta2_mean) / delta2_std
    stacked = [arr.reshape((h, w, 1)) for arr in (base, delta, delta2)]
    return np.concatenate(stacked, axis=2)

# Save each patient's data into HDF5
def save_to_hdf5(numbers, base_path="D:/Data/log_mel_sliding", data_type="log_mel", output_file="test_combined_log_mel_all_patients.h5"):
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
        # Iterate over both patient_features and patient_ids to save them correctly
        for patient_id, features in zip(patient_ids, patient_features):
            print(f"Saving features for patient {patient_id} with shape {features.shape}")
            # Save the features using the actual patient ID
            hdf5_file.create_dataset(f"patient_{patient_id}", data=features)

# numbers = list(range(300, 303))  # Patient IDs
# for test
numbers = [303,304,302,300]
save_to_hdf5(numbers)
# if os.path.exists("D:/Data/log_mel_sliding/combined_log_mel_all_patients.h5"):
#     print(f"{'D:/Data/log_mel_sliding/combined_log_mel_all_patients.h5'} already exists. Skipping save_to_hdf5.")
# else:
#     print(f"{'D:/Data/log_mel_sliding/combined_log_mel_all_patients.h5'} does not exist. Saving data...")
#     numbers = list(range(300, 491))  # Patient IDs
#     save_to_hdf5(numbers)  # Save if the file doesn't exist

original_model=alexnet(pretrained=True)
original_dict = original_model.state_dict()
modifiedAlexNet=modifiedAlexNet(pretrained=False)
modified_model_dict = modifiedAlexNet.state_dict()
pretrained_modified_model_dict = {k: v for k, v in original_dict.items() if k in modified_model_dict}

patient_features = []
with h5py.File('D:/Data/log_mel_sliding/test_combined_log_mel_all_patients.h5', 'r') as hdf5_file:
    patient_group = hdf5_file["patients"]

    # We need to sort because h5py doesnt guarantee that its iterating the patient id;s in the order they are in the group
    sorted_patient_ids = sorted(patient_group.keys(), key=lambda x: int(x.split('_')[1]))

    for patient_id in sorted_patient_ids:
        print(patient_id)
        log_mel = patient_group[patient_id]  # Load the patient's log-mel data from HDF5

        results = []
        # for segment in log_mel:
        #     print(f"Segment shape: {segment.shape}")
        log_mel_spec_3d = np.array([get_3d_spec(segment) for segment in log_mel])
        for segment in log_mel_spec_3d:
            npimg = np.transpose(segment,(2,0,1))
            input_tensor=torch.tensor(npimg, dtype=torch.float)
            input_batch = input_tensor.unsqueeze(0)  # Create mini-batch
            with torch.no_grad():
                output = modifiedAlexNet(input_batch)
                results.append(output)
        features = torch.cat(results, dim=0)
        patient_features.append(features.numpy())
    # feature_patients = np.array(patient_features, dtype=object)

save_features_to_hdf5(patient_features, sorted_patient_ids, 'D:/Data/log_mel_sliding/test_audio_features2_sliding.h5')

# np.save('D:/Data/audio_features2_sliding.npy', feature_patients)
