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


def preprocess_segment(segment,sr):
    # Add noise and pertrube
    alpha = random.uniform(0.01, 0.1)
    random_index = np.random.randint(0, segment.size)
    random_x = segment[random_index]
    perturbed_segment = segment - alpha * random_x

    # Pitch augmentation
    semitones = np.random.uniform(-2, 2)  # Random number in the range [-2, 2] for pitch 
    shifted_segment = librosa.effects.pitch_shift(perturbed_segment, sr=sr, n_steps=semitones)
    return shifted_segment

def preprocessing(audio_segments,sr):
    processed_segments = np.array([preprocess_segment(segment, sr) for segment in audio_segments])
    return processed_segments


def calculate_melspec_for_segments(audio_segments, sr):
    # Initialize parameters for mel-spectrogram calculation
    n_fft = int(0.025 * sr)  # Window length: 25 ms
    hop_length = int(0.010 * sr)  # Hop length: 10 ms
    n_mels = 64  # Number of Mel bands

    log_mel_segments = []

    for segment in audio_segments:
        # Apply STFT
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length, window='hann')
        S = np.abs(stft)**2

        # Convert to Mel scale
        mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)

        # Convert to log scale (add offset to avoid log(0))
        log_mel_S = librosa.power_to_db(mel_S, ref=np.max)
        
        log_mel_segments.append(log_mel_S)

    return log_mel_segments, hop_length


def segment_audio(audio, sr, segment_length_sec=3.5):
    samples_per_segment = int(segment_length_sec * sr)
    total_samples = len(audio)
    
    # Calculate the total number of segments, rounding up to include the last partial segment
    num_segments = int(np.ceil(total_samples / samples_per_segment))
    
    # Initialize an empty list to hold segmented audio
    segments = []
    
    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        
        # Slice the audio array for the current segment
        segment = audio[start_sample:end_sample]
        
        # If the segment is shorter than samples_per_segment, pad it
        if len(segment) < samples_per_segment:
            pad_length = samples_per_segment - len(segment)
            segment = np.pad(segment, (0, pad_length), mode='constant', constant_values=(0, 0))
        
        segments.append(segment)
    
    # Convert the list of segments into a NumPy array
    segments_array = np.array(segments)
    
    return segments_array


# Read the files
file_audios = []
file_visuals = []
base_path = "C:/Users/eleni/Data/"
patient = "_P/"
audio_extension = "_AUDIO.wav"
visual_extension = "_CLNF_AUs.txt"

numbers = [303, 319]
# read files from all patients
for number in numbers:
    file_audio = f"{base_path}{number}{patient}{number}{audio_extension}"
    file_visual = f"{base_path}{number}{patient}{number}{visual_extension}"
    file_audios.append(file_audio)
    file_visuals.append(file_visual)

# extract audios
audios = []
srs = []
preprocessed = []
num_of_segments = []
for file in file_audios:
    audio, sr = librosa.load(file, sr=None)    
    # segment audios
    audio_segments = segment_audio(audio,sr)
    num_of_segments.append(audio_segments.shape[0])
    # Audio preprocessing
    preprocessed_audio_seg = preprocessing(audio_segments,sr)
    preprocessed.append(preprocessed_audio_seg)
    srs.append(sr)

preprocessed_audio_segments = np.array(preprocessed, dtype=object)
# preprocessed_audio_segments = preprocessing(audio_segments,sr)
np.save('C:/Users/eleni/Data/audio_segments.npy', preprocessed_audio_segments)

# Calculate log-mels for the segmented audios
log_mel_segments = []
hop_lengths = []
for i, preprocessed_audio_segment in enumerate(preprocessed_audio_segments):
# for preprocessed_audio_segment in preprocessed_audio_segments:
    # Creating log mel from segmented audio
    log_mel_segment, hop_length = calculate_melspec_for_segments(preprocessed_audio_segment, srs[i])
    log_mel_segment = np.array(log_mel_segment)
    log_mel_segments.append(log_mel_segment)
    hop_lengths.append(hop_length)
log_mel_segments = np.array(log_mel_segments, dtype=object)
np.save('C:/Users/eleni/Data/log_mel.npy', log_mel_segments)



# VISUAL 
# Aligning audio and visual features
visual_frame_rate = 30  # Frames per second, adjust according to your data
audio_segment_duration = 3.5  # Duration of each audio segment in seconds 
visual_frames_per_segment = int(visual_frame_rate * audio_segment_duration)
visuals = []
aggregated_visual_features = []
for i, file_visual in enumerate(file_visuals):
# for file_visual in file_visuals:22
    # Visual preprocessing
    f = open(file_visual, "r")
    # skip first line (title)
    next(f)
    file_visual = f.readlines()
    # convert list of strings into 2D numpy array
    visual_np = [np.fromstring(s, dtype=np.float32, sep=', ') for s in file_visual]
    visual = np.vstack(visual_np)

    # Initialize an array to hold the aggregated visual features for each audio segment
    aggregated_visual = np.zeros((num_of_segments[i], 24))  # 282 segments(for the initial patient), 2560 features per visual frame

    # we are not sure that each patient has 282 segments so needs to be adjusted
    for segment_index in range(num_of_segments[i]):
        # Calculate the start and end frame indices for the visual data corresponding to this audio segment
        start_frame = segment_index * visual_frames_per_segment
        end_frame = start_frame + visual_frames_per_segment
        
        # Ensure the end_frame does not exceed the total number of frames
        end_frame = min(end_frame, visual.shape[0])

        # Aggregate visual features within this segment by calculating the mean
        aggregated_visual[segment_index, :] = visual[start_frame:end_frame].mean(axis=0)
        # size of aggregated visual is (282,24)
    visuals.append(visual)
    aggregated_visual_features.append(aggregated_visual)

visuals = np.array(visuals, dtype=object)
aggregated_visual_features = np.array(aggregated_visual_features, dtype=object)


np.save('C:/Users/eleni/Data/visual.npy', visuals)
# print('Shape of affregated')
# shape is (2,282,24)
# print(aggregated_visual_features.shape)
np.save('C:/Users/eleni/Data/aggr_visual.npy', aggregated_visual_features)

# Labels save
labels_list=[]
with open('C:/Users/eleni/Data/train_split.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(csvfile)
    for row in spamreader:
        labels_list.append(row[1])

labels = np.array(labels_list).astype(np.float32)
np.save('C:/Users/eleni/Data/labels.npy', labels)


segment_index = 15  # Index of the segment you want to visualize
log_mel_S = log_mel_segments[1][segment_index]

# Plot log mel specs
plt.figure(figsize=(10, 4))
librosa.display.specshow(log_mel_S, sr=sr, hop_length=hop_lengths[1], x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Log-Mel Spectrogram for Segment {segment_index}')
plt.tight_layout()
plt.show()