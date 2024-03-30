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
import matplotlib.pyplot as plt

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

def calculate_melspec(preprocessed_audio,sr):
    # calculate mel-spec
    # Parameters
    n_fft = int(0.025 * sr)  # Window length: 25 ms
    hop_length = int(0.010 * sr)  # Hop length: 10 ms
    n_mels = 64  # Number of Mel bands

    # Apply STFT
    stft = librosa.stft(preprocessed_audio, n_fft=n_fft, hop_length=hop_length, window='hann')
    S = np.abs(stft)**2

    # Convert to Mel scale
    mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)

    # Convert to log scale (add offset to avoid log(0))
    log_mel_S = librosa.power_to_db(mel_S, ref=np.max)
    return log_mel_S,hop_length

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

preprocessed_audio = preprocessing(audio)
np.save('C:/Users/eleni/Data/ausio.npy', preprocessed_audio)

labels_list=[]
with open('C:/Users/eleni/Data/train_split.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(csvfile)
    for row in spamreader:
        labels_list.append(row[1])

labels = np.array(labels_list).astype(np.float32)
np.save('C:/Users/eleni/Data/labels.npy', labels)



log_mel_S, hop_length = calculate_melspec(preprocessed_audio,sr)
np.save('C:/Users/eleni/Data/log_mel.npy', log_mel_S)

plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')    
plt.title('Log-Mel spectrogram')
plt.tight_layout()
plt.show()