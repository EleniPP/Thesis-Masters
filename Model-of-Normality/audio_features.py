import numpy as np
import torch 
import torchvision.models as models
import tensorflow as tf
import tensorflow_hub as hub
import torch.nn as nn
import pickle

# Maybe not needed since I already segment in the preoprocessing
def adjust_segments(segments, target_frames=96, num_bands=64):
    adjusted_segments = []
    for segment in segments:
        # Conditionally pad or slice the segment to reach the target frames
        if segment.shape[1] < target_frames:
            pad_width = ((0, 0), (0, target_frames - segment.shape[1]))
            adjusted = np.pad(segment, pad_width=pad_width, mode='constant', constant_values=0)
        elif segment.shape[1] > target_frames:
            adjusted = segment[:, :target_frames]
        else:
            adjusted = segment

        # Correctly reshape for VGGish input without adding an extra dimension
        adjusted = adjusted[np.newaxis, :, :]  # Shape becomes [1, num_bands, num_frames]
        adjusted_segments.append(adjusted)

    # Combine the adjusted segments into a single numpy array
    return np.concatenate(adjusted_segments, axis=0)  # Shape becomes [num_segments, num_bands, nu



log_mel_seg = np.load('C:/Users/eleni/Data/log_mel.npy')
# log_mel_segments is numpy array
# Adjust segments to have the correct shape
log_mel_segments = adjust_segments(log_mel_seg)
# Load the VGGish model
vggish_model = hub.load("https://tfhub.dev/google/vggish/1")

# ----------------------------Not segmented leg-mel-----------------------
# spectrogram_input = np.expand_dims(log_mel, axis=[0, -1]).astype(np.float32)

# # Use VGGish to extract embeddings
# features = vggish_model(spectrogram_input)
# print(type(features))
# # Save a tensor
# with open('C:/Users/eleni/Data/audio_features.pkl', 'wb') as fl:
#     pickle.dump(features, fl)

# -------------------------Segmented log-mel-------------------------------
all_features = []

# Iterate over each log-mel spectrogram segment
for segment in log_mel_segments:
    # Ensure the segment is correctly shaped for VGGish input
    # VGGish expects: [batch_size, num_frames, num_bands, num_channels]
    spectrogram_input = np.expand_dims(segment, axis=0)  # Add batch dimension
    spectrogram_input = np.expand_dims(spectrogram_input, axis=-1)  # Add channel dimension
    spectrogram_input = spectrogram_input.astype(np.float32)

    # Use VGGish to extract embeddings for the current segment
    features = vggish_model(spectrogram_input)
    all_features.append(features)

# Optionally, convert all_features to a numpy array for convenience if needed
all_features_array = np.array(all_features)

# Save the extracted features for all segments
with open('C:/Users/eleni/Data/audio_features.pkl', 'wb') as fl:
    pickle.dump(all_features_array, fl)
