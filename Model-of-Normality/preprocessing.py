import csv
import zipfile 
from scipy.io.wavfile import read
import numpy as np
import wave
import random
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def extract_zip():
    with zipfile.ZipFile("C:/Users/eleni/Data/303_P.zip", 'r') as zip_ref:
        zip_ref.extractall("C:/Users/eleni/Data/303_P")


def preprocessing(audio):
    # Add noise and pertrube
    alpha = random.uniform(0.01, 0.1)
    random_index = np.random.randint(0, audio.size)
    random_x = audio[random_index]
    perturbed_audio = audio - alpha * random_x

    # Pitch augmentation
    semitones = np.random.uniform(-2, 2)  # Random number in the range [-2, 2] for pitch 
    shifted_audio = librosa.effects.pitch_shift(perturbed_audio, sr=sr, n_steps=semitones)
    return shifted_audio

# I dont like it there is a big print and i dont understand what is happening
def align_and_dividing(preprocessed_audio,visual):
    # Align sequences using DTW
    # there is a weird print there for some reason
    distance, path = fastdtw(preprocessed_audio, visual, dist=euclidean)

    # Extract aligned segments
    alignment_indices = [pair[1] for pair in path]
    segment_duration = 3.5  # Segment duration in seconds
    sampling_rate = 44100  # Example sampling rate for audio data (adjust as needed)
    segment_indices = [int(idx * sampling_rate) for idx in alignment_indices]

    # Extract segments of aligned data
    aligned_segments = []
    for idx in segment_indices:
        segment_end = idx + int(segment_duration * sampling_rate)
        aligned_segment = preprocessed_audio[idx:segment_end], visual[path[idx]]
        aligned_segments.append(aligned_segment)
    return aligned_segments


# Read file
file = "C:/Users/eleni/Data/303_P/303_AUDIO.wav"
file_visual = "C:/Users/eleni/Data/303_P/303_CLNF_AUs.txt"
audio, sr = librosa.load(file, sr=None) 

# , errors='ignore'
file_visual = open(file_visual, "r")
visual = file_visual.read()

preprocessed_audio = preprocessing(audio)

aligned_segments = align_and_dividing(preprocessed_audio, visual)
