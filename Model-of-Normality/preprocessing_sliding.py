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
import json
from pprint import pprint
import os

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
    n_fft = int(0.015 * sr)  # Window length: 25 ms
    hop_length = int(0.010 * sr)  # Hop length: 10 ms
    n_mels = 64  # Number of Mel bands

    log_mel_segments = []
    cnt =0
    for segment in audio_segments:
        # Apply STFT
        cnt=cnt+1
        stft = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length, window='hann')
        ft = np.abs(stft)
        # plt.plot(ft)
        # plt.title('Spectrum')
        # plt.xlabel('Frequency Bin')
        # plt.ylabel('Amplitude')
        # plt.show()
        # print(1/0)
        S = np.abs(stft)**2

        # Convert to Mel scale
        mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)

        # Convert to log scale (add offset to avoid log(0))
        log_mel_S = librosa.power_to_db(mel_S, ref=np.max)
        # if cnt==15 or  cnt==125 or cnt==200:
        #     plt.figure(figsize=(10, 4))
        #     librosa.display.specshow(log_mel_S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        #     plt.colorbar(format='%+2.0f dB')
        #     plt.title(f'Log-Mel Spectrogram for Segment 1')
        #     plt.tight_layout()
        #     plt.show()
        #     if cnt ==200:
        #         print(1/0)  
        log_mel_segments.append(log_mel_S)

    return log_mel_segments, hop_length


def segment_audio(audio, sr, segment_length_sec=3.5, step_length_sec=0.1):
    samples_per_segment = int(segment_length_sec * sr)
    step_size = int(step_length_sec * sr)  # Step size in samples
    total_samples = len(audio)
    
    # Calculate the total number of segments, rounding up to include the last partial segment
    # num_segments = int(np.ceil(total_samples / samples_per_segment))
    
    # Initialize an empty list to hold segmented audio
    segments = []
    
   # Start with the first sample and slide with the step size
    start_sample = 0
    while start_sample < total_samples:
        end_sample = start_sample + samples_per_segment
        
        # Slice the audio array for the current segment
        segment = audio[start_sample:end_sample]
        
        # If the segment is shorter than samples_per_segment, pad it
        if len(segment) < samples_per_segment:
            pad_length = samples_per_segment - len(segment)
            segment = np.pad(segment, (0, pad_length), mode='constant', constant_values=(0, 0))
        
        segments.append(segment)
        
        # Move the window forward by the step size (not the full segment length)
        start_sample += step_size
    
    # Convert the list of segments into a NumPy array
    segments_array = np.array(segments)
    
    return segments_array


def get_labels(split):
    base_path = "V:/staff-umbrella/EleniSalient/"
    extension = "_split.csv"
    file = f"{base_path}{split}{extension}"
    labels_list=[]
    patients_list=[]
    max_row = 47
    # with open('C:/Users/eleni/Data/train_split.csv', newline='') as csvfile:
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        row_count = 0
        for row in spamreader:
            if (row_count >= max_row) and (split == "1"):
                break
            labels_list.append(row[1])
            patients_list.append(row[0])
            row_count += 1

    # Convert lists to numpy arrays
    patients_array = np.array(patients_list).astype(np.int32)
    labels_array = np.array(labels_list).astype(np.float32)

    # Combine the arrays into a single array with shape (107, 2)
    combined_array = np.column_stack((patients_array, labels_array))
    return combined_array

# # Train split
# train = get_labels("train")
# print(train.shape)
# # Validation split
# val = get_labels("dev")
# print(val.shape)
# # Test split
# test = get_labels("test")
# print(test.shape)

# concatenated_array = np.concatenate((train, test, val), axis=0)

# # Sort the concatenated array based on the first column (patient number)
# labels_array = concatenated_array[concatenated_array[:, 0].argsort()]

# labels_dict = {}

# # Iterate over each row in the array
# for row in labels_array:
#     key = int(row[0])
#     value = int(row[1])
#     labels_dict[key] = value

# # This has all the labels in the order of the patients
# np.save('V:/staff-umbrella/EleniSalient/Data/labels.npy', labels_array)


# Read the files
# base_path = "C:/Users/eleni/Data/"
base_path = "V:/staff-umbrella/EleniSalient/"
patient = "_P/"
audio_extension = "_AUDIO.wav"
visual_extension = "_CLNF_AUs.txt"

# numbers = [303, 319]
# numbers = list(range(300, 492))
numbers = list(range(367, 492))
# read files from all patients
srs = []
preprocessed = []
num_of_segments = []
# log_mel_segments = []
# hop_lengths = []
visuals = []
# aggregated_visual_features = []
for number in numbers:
    audio_ssd = f'D:/Data/audio_{number}.npy'
    visual_ssd = f'D:/Data/visual_{number}.npy'
    file_audio = f"{base_path}{number}{patient}{number}{audio_extension}"
    file_visual = f"{base_path}{number}{patient}{number}{visual_extension}"

    # EXTRACT AUDIO
      # Check if audio file already exists
    if os.path.exists(audio_ssd):
        print(f"Audio file for patient {number} already exists. Loading from disk.")
        audio = np.load(audio_ssd)
        sr = 16000
    else:
        # EXTRACT AUDIO
        try:
            audio, sr = librosa.load(file_audio, sr=None)   #sampling rate at 16000 (16kHz) like in the dataset
            np.save(f'D:/Data/audio_{number}.npy', audio)
            # sr = 16000
            # plt.plot(audio)
            # plt.title('Signal')
            # plt.xlabel('Time (samples)')
            # plt.ylabel('Amplitude')x
            # plt.show()
            # print(1/0)
        except FileNotFoundError as e:
            print(f"Audio file not found for number {number}: {e}")
            # if number in labels_dict:
            #     del labels_dict[number]  # Remove the entry from the dictionary
            continue  # Skip to the next iteration if the audio file is not found
    # Segment audios
    audio_segments = segment_audio(audio,sr)
    number_of_segments = audio_segments.shape[0]

    # Audio preprocessing
    preprocessed_audio_segment = preprocessing(audio_segments,sr)

    # Creating log mel from segmented audio
    log_mel_segment, hop_length = calculate_melspec_for_segments(preprocessed_audio_segment, sr)
    log_mel_segment = np.array(log_mel_segment)


    # log_mel_segments.append(log_mel_segment)
    # hop_lengths.append(hop_length) #its for the plot

    # VISUAL 
    
    visual_frame_rate = 30  # Frames per second, adjust according to your data
    audio_segment_duration = 3.5  # Duration of each audio segment in seconds 
    visual_frames_per_segment = int(visual_frame_rate * audio_segment_duration)
    step_size_sec = 0.1  # Step size in seconds (for audio sliding window)
    step_size_frames = int(visual_frame_rate * step_size_sec)  # Number of visual frames corresponding to the step size (0.1 sec)

    # Visual preprocessing

    if os.path.exists(visual_ssd):
        print(f"Visual file for patient {number} already exists. Loading from disk.")
        visual = np.load(visual_ssd)
    else:
        # EXTRACT VISUAL FEATURES
        try:
            with open(file_visual, "r") as f:
                # Skip first line (title)
                next(f)
                file_visual = f.readlines()
                                # Convert list of strings into 2D numpy array
                visual_np = [np.fromstring(s, dtype=np.float32, sep=', ') for s in file_visual]
                visual = np.vstack(visual_np)
                np.save(f'D:/Data/visual_{number}.npy', visual)
        except FileNotFoundError as e:
            print(f"Visual file not found for number {number}: {e}")
            continue  # Skip to the next iteration if the audio file is not found

    # Initialize an array to hold the aggregated visual features for each audio segment
    aggregated_visual = np.zeros((number_of_segments, 24))  # 282 segments(for the initial patient), 2560 features per visual frame
    
    # Aligning audio and visual features
    start_frame = 0
    for segment_index in range(number_of_segments):
    # Calculate the end frame for the current audio segment
        end_frame = start_frame + visual_frames_per_segment
        
        # Ensure the end_frame does not exceed the total number of frames
        end_frame = min(end_frame, visual.shape[0])
        
        # Aggregate visual features within this segment by calculating the mean
        aggregated_visual[segment_index, :] = visual[start_frame:end_frame].mean(axis=0)
        
        # Move the window forward by the step size (not by the full segment length)
        start_frame += step_size_frames
    # aggregated_visual_features.append(aggregated_visual)
    aggregated_visual = np.array(aggregated_visual)
    np.save(f'D:/Data/log_mel_sliding/log_mel_{number}.npy', log_mel_segment)
    np.save(f'D:/Data/aggr_visual_sliding/visual_{number}.npy', aggregated_visual)
    print("Done with:", number)

# Save arrays in files
# log_mel_segments = np.array(log_mel_segments, dtype=object)
# np.save('D:/Data/log_mel_sliding.npy', log_mel_segments)_
# aggregated_visual_features = np.array(aggregated_visual_features, dtype=object)
# np.save('D:/Data/aggr_visual_sliding.npy', aggregated_visual_features)

# print("lenght:", len(labels_dict))
# pprint(labels_dict)
# with open('V:/staff-umbrella/EleniSalient/Data/labels.json', 'w') as file:
#     json.dump(labels_dict, file)




# --------------------------------------------------------test----------------------------------
# Simulate an audio signal with 10.5 seconds of data
# sr = 16000  # Sample rate (16kHz)
# audio_length_sec = 10.5  # Total audio length in seconds
# audio = np.random.rand(int(audio_length_sec * sr))  # Random audio data

# # Check the segmenting function
# segments = segment_audio(audio, sr, segment_length_sec=3.5)

# # Expected result: 10.5 sec / 3.5 sec segments = 3 segments
# # Total samples per segment = 3.5 sec * 16,000 samples/sec = 56,000 samples per segment
# print(f"Total segments: {segments.shape[0]}")
# print(f"Segment shape (each segment should be 56,000 samples): {segments.shape[1]}")

# ----------------------------------------------------------------------------------------------


# segment_index = 15  # Index of the segment you want to visualize
# log_mel_S = log_mel_segments[1][segment_index]

# # Plot log mel specs
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(log_mel_S, sr=sr, hop_length=hop_lengths[1], x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title(f'Log-Mel Spectrogram for Segment {segment_index}')
# plt.tight_layout()
# plt.show()