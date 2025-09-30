import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import os

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

N_CHANNELS = 3

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=512):
        super(ModifiedAlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # here the input 3 is for an image with 3 channels
            # I gave one channel which is the log mel spectrogram / also kernel size,sride and padding are tuples because we have 2 spatial dimensions which are mel and frames
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

        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

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
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
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


if __name__ == "__main__":
    # Load logmel data
    log_mel_data = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/log_mels_reliable_prep.npy', allow_pickle=True)
    modified_model = modifiedAlexNet(pretrained=False)
    modified_model_dict = modified_model.state_dict()
    # pretrained_modified_model_dict = {k: v for k, v in original_dict.items() if k in modified_model_dict}

    patient_features = []

    for patient_idx, log_mel in enumerate(log_mel_data):
        print(f"Processing patient {patient_idx}")
        results = []
        log_mel_spec_3d = np.array([get_3d_spec(segment) for segment in log_mel])
        for segment in log_mel_spec_3d:
            npimg = np.transpose(segment, (2, 0, 1))
            # npimg = np.expand_dims(segment, axis=0)
            input_tensor = torch.tensor(npimg, dtype=torch.float)
            input_batch = input_tensor.unsqueeze(0)  # Create mini-batch
            with torch.no_grad():
                output = modified_model(input_batch)
                results.append(output)
        features = torch.cat(results, dim=0)
        # Append features to the patient features list
        patient_features.append(features.numpy())
        if isinstance(log_mel, np.ndarray):
            num_segments = log_mel.shape[0]  # Use shape if it's a numpy array
        else:
            num_segments = len(log_mel)  # Use len() if it's a list
        print(f"Patient {patient_idx}: {num_segments} segments")

    feature_patients = np.array(patient_features, dtype=object)
    # Save features to .npy file
    np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/audio_features_reduced_reliable_prep.npy', feature_patients)
    print("Features saved to audio_features.npy")