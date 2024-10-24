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
import pandas as pd
import soundfile as sf

def extract_zip():
    with zipfile.ZipFile("C:/Users/eleni/Data/303_P.zip", 'r') as zip_ref:
        zip_ref.extractall("C:/Users/eleni/Data/303_P")


def preprocess_audio(audio,sr):
    # Add noise and pertrube
    alpha = random.uniform(0.01, 0.1)
    random_index = np.random.randint(0, audio.size)
    random_x = audio[random_index]
    perturbed_audio = audio - alpha * random_x

    # Pitch augmentation
    semitones = np.random.uniform(-2, 2)  # Random number in the range [-2, 2] for pitch 
    shifted_audio = librosa.effects.pitch_shift(perturbed_audio, sr=sr, n_steps=semitones)
    return shifted_audio


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
            if (row_count >= max_row) and (split == "test"):
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

# Train split
train = get_labels("train")
print(train.shape)
# Validation split
val = get_labels("dev")
print(val.shape)
# Test split
test = get_labels("test")
print(test.shape)

concatenated_array = np.concatenate((train, test, val), axis=0)

# Sort the concatenated array based on the first column (patient number)
labels_array = concatenated_array[concatenated_array[:, 0].argsort()]

labels_dict = {}

# Iterate over each row in the array
for row in labels_array:
    key = int(row[0])
    value = int(row[1])
    labels_dict[key] = value


# # This has all the labels in the order of the patients
# np.save('V:/staff-umbrella/EleniSalient/Data/labels.npy', labels_array)


# Read the files
# base_path = "C:/Users/eleni/Data/"
base_path = "V:/staff-umbrella/EleniSalient/"
patient = "_P/"
audio_extension = "_AUDIO.wav"
visual_extension = "_CLNF_AUs.txt"
transcript_extension = "_TRANSCRIPT.csv"

# numbers = [303, 319]
numbers = list(range(300, 492))
# read files from all patients
srs = []
preprocessed = []
num_of_segments = []
log_mel_segments = []
hop_lengths = []
visuals = []
aggregated_visual_features = []
for number in numbers:
    file_audio = f"{base_path}{number}{patient}{number}{audio_extension}"
    file_visual = f"{base_path}{number}{patient}{number}{visual_extension}"
    transcript_file = f"{base_path}{number}{patient}{number}{transcript_extension}"

    # EXTRACT AUDIO
    try:
        audio, sr = librosa.load(file_audio, sr=None)    
    except FileNotFoundError as e:
        print(f"Audio file not found for number {number}: {e}")
        if number in labels_dict:
            del labels_dict[number]  # Remove the entry from the dictionary
        continue  # Skip to the next iteration if the audio file is not found

    print(audio.shape)
    # Load the transcript CSV file
    transcript_df = pd.read_csv(transcript_file, sep='\t')

    # Create an array of zeros (silence) the same size as the full audio
    final_audio = np.zeros_like(audio)
    print(final_audio.shape)
    # Initialize an empty list to store the participant's audio segments
    participant_segments = []
    # Iterate through the transcript to segment the audio
    for index, row in transcript_df.iterrows():
        start_time = row['start_time']
        stop_time = row['stop_time']
        speaker = row['speaker']
        text = row['value']
        
        # Skip entries marked as scrubbed
        if "scrubbed_entry" in text.lower():
            print(f"Skipping scrubbed entry from {start_time} to {stop_time}")
            continue

        # Only process segments where the speaker is the participant
        if speaker.lower() == 'participant':
            # Convert start and stop times from seconds to sample indices
            start_sample = int(start_time * sr)
            stop_sample = int(stop_time * sr)

            # Extract the audio segment
            audio_segment = audio[start_sample:stop_sample]
            
            # Append the participant's audio segment to the list
            participant_segments.append(audio_segment)

            # Copy the participant's speech into the final audio array
            final_audio[start_sample:stop_sample] = audio_segment

    # preprocessed_audio = preprocess_audio(final_audio,sr)

    # Step 2: Calculate Mel-spectrogram for the final_audio
    n_fft = int(0.025 * sr)  # Window length: 25 ms
    hop_length = int(0.010 * sr)  # Hop length: 10 ms
    n_mels = 64  # Number of Mel bands

    stft = librosa.stft(final_audio, n_fft=n_fft, hop_length=hop_length, window='hann')
    S = np.abs(stft)**2
    # Convert to Mel scale
    mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)

    # Convert to log scale (add offset to avoid log(0))
    log_mel_spectrogram = librosa.power_to_db(mel_S, ref=np.max)

    # SEGMENT LOG MEL
    # Parameters for segmenting
    segment_duration = 3.5  # in seconds
    stride_duration = 0.1  # in seconds

    frames_per_segment = int(segment_duration * sr / hop_length)  # 3.5 seconds worth of frames
    frames_per_stride = int(stride_duration * sr / hop_length)    # 0.1 seconds worth of frames

    mel_segments = []
    start_frame = 0

    while start_frame + frames_per_segment <= log_mel_spectrogram.shape[1]:
        end_frame = start_frame + frames_per_segment
        
        # Extract the Mel-spectrogram for the current window
        mel_segment = log_mel_spectrogram[:, start_frame:end_frame]
        mel_segments.append(mel_segment)
        
        # Move the window by the stride (0.1 seconds worth of frames)
        start_frame += frames_per_stride

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_segments[620], sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram for 1st 3.5s Segment (Sliding Window)')
    plt.tight_layout()
    plt.show()

    # Step 5: Check number of segments and shapes
    print(f"Number of segments: {len(mel_segments)}")
    print(f"Shape of one Mel-spectrogram segment: {mel_segments[0].shape}")  # Should be (n_mels, frames_per_segment)

    # Visual
    fps_au = 30  # 30 AU frames per second
    frames_per_segment_au = int(segment_duration * fps_au)  # 3.5 seconds worth of AU frames (105 frames)
    frames_per_stride_au = int(stride_duration * fps_au)    # 0.1 seconds worth of AU frames (3 frames)

    au_df = pd.read_csv(file_visual, sep=', ', engine='python')

    # Make sure they are the same length audio and visual
    total_audio_duration = len(audio) / sr  # Total duration of audio in seconds
    total_au_duration = au_df['timestamp'].iloc[-1]  # Last AU timestamp

        # Ensure both modalities are aligned by trimming the longer one
    if total_audio_duration > total_au_duration:
        # Trim the audio to match the AU duration
        num_samples = int(total_au_duration * sr)
        audio = audio[:num_samples]
    elif total_au_duration > total_audio_duration:
        # Trim the AU data to match the audio duration
        au_df = au_df[au_df['timestamp'] <= total_audio_duration]


    #  Step 1: Extract the AU features (ignoring metadata like frame, timestamp, confidence, success)
    au_features = au_df.drop(['frame', 'timestamp', 'confidence', 'success'], axis=1)

    # Step 2: Sliding window segmentation of AUs
    au_segments = []
    start_frame = 0

    # Sliding window over the AU frames
    while start_frame + frames_per_segment_au <= len(au_features):
        end_frame = start_frame + frames_per_segment_au
        
        # Extract AU frames for the current window
        au_segment = au_features.iloc[start_frame:end_frame].values
        au_segments.append(au_segment)
        
        # Move the window by the stride (0.1 seconds worth of frames)
        start_frame += frames_per_stride_au

    # Step 3: Check number of AU segments and shapes
    print(f"Number of AU segments: {len(au_segments)}")
    print(f"Shape of one AU segment: {au_segments[0].shape}")  # Should be (105, num_AUs)
    print(1/0)


#     # Initialize an array to hold the aggregated visual features for each audio segment
#     aggregated_visual = np.zeros((number_of_segments, 24))  # 282 segments(for the initial patient), 2560 features per visual frame

#     for segment_index in range(number_of_segments):
#         # Calculate the start and end frame indices for the visual data corresponding to this audio segment
#         start_frame = segment_index * visual_frames_per_segment
#         end_frame = start_frame + visual_frames_per_segment
        
#   x      # Ensure the end_frame does not exceed the total number of frames
#         end_frame = min(end_frame, visual.shape[0])

#         # Aggregate visual features within this segment by calculating the mean
#         aggregated_visual[segment_index, :] = visual[start_frame:end_frame].mean(axis=0)
#     aggregated_visual_features.append(aggregated_visual)
#     print("Done with:", number)

# # Save arrays in files
# log_mel_segments = np.array(log_mel_segments, dtype=object)
# np.save('V:/staff-umbrella/EleniSalient/Data/log_mel.npy', log_mel_segments)
# aggregated_visual_features = np.array(aggregated_visual_features, dtype=object)
# np.save('V:/staff-umbrella/EleniSalient/Data/aggr_visual.npy', aggregated_visual_features)

# print("lenght:", len(labels_dict))
# pprint(labels_dict)
# with open('V:/staff-umbrella/EleniSalient/Data/labels.json', 'w') as file:
#     json.dump(labels_dict, file)
