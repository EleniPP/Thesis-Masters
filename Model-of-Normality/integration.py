import torch
import pickle

# Load the tensor
with open('C:/Users/eleni/Data/audio_features.pkl', 'rb') as f:
    audio_features = pickle.load(f)


# Load the tensor
with open('C:/Users/eleni/Data/visual_features.pkl', 'rb') as f:
    visual_features = pickle.load(f)


print(audio_features.size())
print(visual_features.size())