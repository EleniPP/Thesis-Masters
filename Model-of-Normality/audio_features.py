import numpy as np
import torch 
import torchvision.models as models
import tensorflow as tf
import tensorflow_hub as hub
import torch.nn as nn
import pickle

log_mel = np.load('C:/Users/eleni/Data/log_mel.npy')

# Load the VGGish model
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")

spectrogram_input = np.expand_dims(log_mel, axis=[0, -1]).astype(np.float32)

# Use VGGish to extract embeddings
features = vggish_model(spectrogram_input)

# Save a tensor
with open('C:/Users/eleni/Data/audio_features.pkl', 'wb') as f:
    pickle.dump(features, f)
