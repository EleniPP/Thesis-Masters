import csv
import zipfile 
from scipy.io.wavfile import read
import numpy as np
import wave
import random
import librosa



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

# Read file
file = "C:/Users/eleni/Data/303_P/303_AUDIO.wav"
visual = "C:/Users/eleni/Data/303_P/303_CLNF_AUs.txt"
audio, sr = librosa.load(file, sr=None) 

preprocessed_audio = preprocessing(audio)
