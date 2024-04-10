import torch
import pickle

# Load the tensor
with open('C:/Users/eleni/Data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)


# Load the tensor
with open('C:/Users/eleni/Data/visual_features2.pkl', 'rb') as f:
    visual_features = pickle.load(f)

multimodal = torch.cat((audio_features, visual_features), dim=1)


print(audio_features.shape)
print(visual_features.shape)