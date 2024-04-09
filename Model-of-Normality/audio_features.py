import numpy as np
import torch 
import torchvision.models as models
import tensorflow as tf
import tensorflow_hub as hub
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}



log_mel_seg = np.load('C:/Users/eleni/Data/log_mel.npy')
# log_mel_segments is numpy array

# Load the VGGish model
# vggish_model = hub.load("https://tfhub.dev/google/vggish/1")

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
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
     

class ModifiedAlexNet(nn.Module):
    def __init__(self, num_classes=4):
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
        #print(x.shape)
        x=self.classifier(x)
        #print('classifier',x)
        #x=self.softmax(x)
        #print('softmax',x)
        #x = self.avgpool(x)
        #print('avgpool',x.shape)
        #x = torch.flatten(x, 1)
        #print('flatten',x.shape)
        #x = self.classifier(x)
        return x
   
def modifiedAlexNet(pretrained=False, progress=True, **kwargs):
    model_modified = ModifiedAlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
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

# EXTACT THE MFCC FEATURE USING LIBROSA

spector=get_3d_spec(log_mel_seg)
npimg = np.transpose(spector,(2,0,1))
input_tensor=torch.tensor(npimg)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

original_model=alexnet(pretrained=True)
original_dict = original_model.state_dict()
modifiedAlexNet=modifiedAlexNet(pretrained=False)
modified_model_dict = modifiedAlexNet.state_dict()
pretrained_modified_model_dict = {k: v for k, v in original_dict.items() if k in modified_model_dict}
modifiedAlexNet.to('cuda')


x=get_3d_spec(log_mel_seg)
npimg = np.transpose(x,(2,0,1))
input_tensor=torch.tensor(npimg)

input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    modifiedAlexNet.to('cuda')
with torch.no_grad():
    output = modifiedAlexNet(input_batch)
    #output.squeeze().shape
    #output=torch.flatten(output, start_dim=2)
    #print(output.shape)
    #output=torch.sum(output, dim=2)
    print(output)

# ----------------------------Not segmented leg-mel/ Doesnt work-----------------------
# spectrogram_input = np.expand_dims(log_mel, axis=[0, -1]).astype(np.float32)

# # Use VGGish to extract embeddings
# features = vggish_model(spectrogram_input)
# print(type(features))
# # Save a tensor
# with open('C:/Users/eleni/Data/audio_features.pkl', 'wb') as fl:
#     pickle.dump(features, fl)

# -------------------------Segmented log-mel/ Doesnt work-------------------------------
# all_features = []

# # Iterate over each log-mel spectrogram segment
# for segment in log_mel_segments:
#     # Ensure the segment is correctly shaped for VGGish input
#     # VGGish expects: [batch_size, num_frames, num_bands, num_channels]
#     spectrogram_input = np.expand_dims(segment, axis=0)  # Add batch dimension
#     spectrogram_input = np.expand_dims(spectrogram_input, axis=-1)  # Add channel dimension
#     spectrogram_input = spectrogram_input.astype(np.float32)

#     # Use VGGish to extract embeddings for the current segment
#     features = vggish_model(spectrogram_input)
#     all_features.append(features)

# # Optionally, convert all_features to a numpy array for convenience if needed
# all_features_array = np.array(all_features)

# # Save the extracted features for all segments
# with open('C:/Users/eleni/Data/audio_features.pkl', 'wb') as fl:
#     pickle.dump(all_features_array, fl)


# -----------------------------------The simple one-------------------------------------------------------------
# class AudioFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(AudioFeatureExtractor, self).__init__()
#         # Adjusted kernel sizes and added stride for processing larger frame dimension (351)
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(1, 2), padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 5), stride=(1, 2), padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)


#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         return x

# model = AudioFeatureExtractor()  
# segmented_log_mels_tensor = torch.tensor(log_mel_seg).unsqueeze(1)  
# features = model(segmented_log_mels_tensor)
# print(features)
# print(features.shape)