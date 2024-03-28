import csv
import zipfile 
from scipy.io.wavfile import read
import numpy as np
import wave
import random
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
import csv

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
file_visual = "C:/Users/eleni/Data/303_P/303_CLNF_AUs.txt"
audio, sr = librosa.load(file, sr=None) 

# , errors='ignore'
f = open(file_visual, "r")
# skip first line (title)
next(f)
file_visual = f.readlines()
# convert list of strings into 2D numpy array
visual_np = [np.fromstring(s, dtype=np.float32, sep=', ') for s in file_visual]
visual = np.vstack(visual_np)

np.save('C:/Users/eleni/Data/visual.npy', visual)

# preprocessed_audio = preprocessing(audio)
labels_list=[]
with open('C:/Users/eleni/Data/train_split.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(csvfile)
    for row in spamreader:
        labels_list.append(row[1])

labels = np.array(labels_list).astype(np.float32)
np.save('C:/Users/eleni/Data/labels.npy', labels)
# aligned_segments = align_and_dividing(preprocessed_audio, visual)
