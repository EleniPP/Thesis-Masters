import csv
import numpy as np
import random
import librosa
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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
    base_path = "/tudelft.net/staff-umbrella/EleniSalient/"
    # base_path = "V:/staff-umbrella/EleniSalient/"
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
base_path = "/tudelft.net/staff-umbrella/EleniSalient/"
# base_path = "V:/staff-umbrella/EleniSalient/"
patient = "_P/"
audio_extension = "_AUDIO.wav"
visual_extension = "_CLNF_AUs.txt"
transcript_extension = "_TRANSCRIPT.csv"

# numbers = [303, 319]
numbers = list(range(300, 492))
# read files from all patients
log_mels = []
aus = []
masks = []
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
    
    # print('Shape of audio:')
    # print(audio.shape)
    # Load the transcript CSV file
    transcript_df = pd.read_csv(transcript_file, sep='\t')

    # Create an array of zeros (silence) the same size as the full audio
    final_audio = np.zeros_like(audio)
    
    # Iterate through the transcript to segment the audio
    for index, row in transcript_df.iterrows():
        start_time = row['start_time']
        stop_time = row['stop_time']
        speaker = row['speaker']
        text = row['value']
        
        # Skip entries marked as scrubbed
        if "scrubbed_entry" in str(text).lower():
            print(f"Skipping scrubbed entry from {start_time} to {stop_time}")
            continue

        # Only process segments where the speaker is the participant
        if speaker.lower() == 'participant':
            # Convert start and stop times from seconds to sample indices
            start_sample = int(start_time * sr)
            stop_sample = int(stop_time * sr)

            # Extract the audio segment
            audio_segment = audio[start_sample:stop_sample]

            # Copy the participant's speech into the final audio array
            final_audio[start_sample:stop_sample] = audio_segment

    # TODO: check it out
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

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mel_segments[620], sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Log-Mel Spectrogram for 1st 3.5s Segment (Sliding Window)')
    # plt.tight_layout()
    # plt.show()
    log_mels.append(mel_segments)
    # Step 5: Check number of segments and shapes
    # print(f"Number of segments: {len(mel_segments)}")
    # print(f"Shape of one Mel-spectrogram segment: {mel_segments[0].shape}")  # Should be (n_mels, frames_per_segment) / it is (64,350)

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

    # ZERO out the unreliable data
    au_columns = au_df.columns[4:]  # Assuming the first four columns are metadata
    # Create a binary mask for low confidence and success == 0
    mask = (au_df['confidence'] >= 0.7) & (au_df['success'] == 1)
    # Apply the mask to set the values to zero for unreliable rows
    au_df.loc[~mask, au_columns] = 0  # ~mask inverts the mask, selecting rows with success == 0
    mask = mask.astype(int)  # Convert to integer (1 for reliable, 0 for unreliable)

    #  Step 1: Extract the AU features (ignoring metadata like frame, timestamp, confidence, success)
    au_features = au_df.drop(['frame', 'timestamp', 'confidence', 'success'], axis=1)

    # Extract regression and binary AUs
    au_features_r = au_features.filter(regex='_r$')  # Select only regression outputs (continuous values)
    # DEBUG
    non_numeric_r = au_features_r.apply(lambda x: pd.to_numeric(x, errors='coerce').isna())
    if non_numeric_r.any().any():
        print("Non-numeric or NaN values found in au_features_r at:")
        print(non_numeric_r[non_numeric_r].stack())

    au_features_c = au_features.filter(regex='_c$')  # Select only classification labels (binary 0/1)
    # DEBUG
    non_numeric_c = au_features_c.apply(lambda x: pd.to_numeric(x, errors='coerce').isna())
    if non_numeric_c.any().any():
        print("Non-numeric or NaN values found in au_features_c at:")
        print(non_numeric_c[non_numeric_c].stack())

    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize to [0, 1]
    au_features_r_normalized = pd.DataFrame(scaler.fit_transform(au_features_r), columns=au_features_r.columns)

    # DEBUG
    non_numeric_c = au_features_r_normalized.apply(lambda x: pd.to_numeric(x, errors='coerce').isna())
    if non_numeric_c.any().any():
        print("Non-numeric or NaN values found in au_features_r normalized after MinMax scaler at:")
        print(non_numeric_c[non_numeric_c].stack())

    # Combine both standardized regression and binary features into one dataset
    combined_au_features = pd.concat([au_features_r_normalized, au_features_c], axis=1)
    # DEBUG
    non_numeric_combined = combined_au_features.apply(lambda x: pd.to_numeric(x, errors='coerce').isna())
    if non_numeric_combined.any().any():
        print("Non-numeric or NaN values found in combined_au_features right after combining:")
        print(non_numeric_combined[non_numeric_combined].stack())


    # Reshape the mask to match segment dimensions
    mask_segments = []
    start_frame = 0

    while start_frame + frames_per_segment_au <= len(mask):
        end_frame = start_frame + frames_per_segment_au
        mask_segment = mask.iloc[start_frame:end_frame].values
        mask_segments.append(mask_segment)
        start_frame += frames_per_stride_au

    masks.append(mask_segments)

    # Sliding window segmentation of AUs
    au_segments = []
    start_frame = 0

    # Sliding window over the AU frames
    while start_frame + frames_per_segment_au <= len(combined_au_features):
        end_frame = start_frame + frames_per_segment_au
        
        # Extract AU frames for the current window
        au_segment = combined_au_features.iloc[start_frame:end_frame].values    
        if not np.isfinite(au_segment).all():
            print(f"Non-numeric values detected in `au_segment` for patient {number} in frames {start_frame}:{end_frame}")
        au_segments.append(au_segment)
        
        # Move the window by the stride (0.1 seconds worth of frames)
        start_frame += frames_per_stride_au

    aus.append(au_segments)
    # Step 3: Check number of AU segments and shapes
    # print(f"Number of AU segments: {len(au_segments)}")
    # print(f"Shape of one AU segment: {au_segments[0].shape}")  # Should be (105, num_AUs) / it is (105,20)
    # print(f"Shape of one mask segment: {mask_segments[0].shape}")

# Save arrays in files
log_mels = np.array(log_mels, dtype=object)

# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/log_mels.npy', log_mels)
# np.save('V:/staff-umbrella/EleniSalient/Preprocessing/log_mels.npy', log_mels)

# Convert each segment within each patient to float32
# for patient_index, patient_segments in enumerate(aus):
#     aus[patient_index] = [seg.astype(np.float32) for seg in patient_segments]

# DEBUG
# Iterate through each patient in `aus`
for patient_idx, patient_segments in enumerate(aus):
    # Iterate through each segment for the current patient
    for segment_idx, segment in enumerate(patient_segments):
        # Check if all values in the segment are finite (numeric)
        if not np.isfinite(segment).all():
            print(f"Non-numeric values detected in patient {patient_idx}, segment {segment_idx}")
            # Optionally, print the non-numeric values for more detail
            print(segment[~np.isfinite(segment)])

aus = np.array(aus, dtype=object)

# DEBUG

# Ensure aus is converted to a DataFrame for easier handling of non-numeric values
aus_df = pd.DataFrame(aus)

# Apply pd.to_numeric to coerce non-numeric values to NaN
aus_numeric = aus_df.apply(pd.to_numeric, errors='coerce')

# Convert back to NumPy if needed
aus_numeric_array = aus_numeric.to_numpy()

# Create a mask for NaN values
non_numeric_mask = np.isnan(aus_numeric_array)

# DEBUG: Check if any NaNs are present
if np.any(non_numeric_mask):
    print("Non-numeric or NaN values found in aus at indices:")
    print(np.argwhere(non_numeric_mask))

# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/aus.npy', aus)
# # np.save('V:/staff-umbrella/EleniSalient/Preprocessing/aus.npy', aus)

masks = np.array(masks, dtype=object)
# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/mask_segments.npy', masks)
# np.save('V:/staff-umbrella/EleniSalient/Preprocessing/mask_segments.npy', masks)

print('Log-mels')
print(log_mels.shape)
print(len(log_mels[0]))
print(len(log_mels[1]))

print('Action Units')
print(aus.shape)
print(len(aus[0]))
print(len(aus[1]))

print('Masks')
print(masks.shape)
print(len(masks[0]))
print(len(masks[1]))

# print("lenght:", len(labels_dict))
# pprint(labels_dict)
# with open('V:/staff-umbrella/EleniSalient/Data/labels.json', 'w') as file:
#     json.dump(labels_dict, file)
