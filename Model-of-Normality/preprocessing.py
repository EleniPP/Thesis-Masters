import csv
import zipfile 
from scipy.io.wavfile import read
import numpy as np
import wave
import random


def extract_zip():
    with zipfile.ZipFile("C:/Users/eleni/Data/303_P.zip", 'r') as zip_ref:
        zip_ref.extractall("C:/Users/eleni/Data/303_P")

def read_audio():
    ifile = wave.open("C:/Users/eleni/Data/303_P/303_AUDIO.wav")
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)

    # Convert buffer to float32 using NumPy                                                                                 
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio = audio_as_np_int16.astype(np.float32)

    # Normalise float32 array so that values are between -1.0 and +1.0                                                      
    # max_int16 = 2**15
    # audio_normalised = audio / max_int16
    return audio
  
# TODO try librosa
audio = read_audio()
print(audio.size) 

alpha = random.uniform(0.01, 0.1)
random_index = np.random.randint(0, audio.size)
random_x = audio[random_index]

perturbed_signal = audio - alpha * random_x
print(perturbed_signal)
# with open(file, newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#     for row in spamreader:
#         print(', '.join(row))
