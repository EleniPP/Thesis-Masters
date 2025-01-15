import csv
import numpy as np
import random
import librosa
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from scipy.io import wavfile


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
final_audio_folder = "/tudelft.net/staff-umbrella/EleniSalient/Final_audio/"
# base_path = "V:/staff-umbrella/EleniSalient/"
patient = "_P/"
audio_extension = "_AUDIO.wav"
visual_extension = "_CLNF_AUs.txt"
transcript_extension = "_TRANSCRIPT.csv"
final_audio_extension = "_final_audio.wav"

# numbers = [303, 319]
numbers = list(range(300, 491))
# read files from all patients
log_mels = []
aus = []
masks = []
aus_reliable = []
log_mels_reliable = []
for number in numbers:
    file_audio = f"{base_path}{number}{patient}{number}{audio_extension}"
    file_visual = f"{base_path}{number}{patient}{number}{visual_extension}"
    transcript_file = f"{base_path}{number}{patient}{number}{transcript_extension}"
    final_audio_file = f"{final_audio_folder}{number}{final_audio_extension}"

    # EXTRACT AUDIO
    try:
        audio, sr = librosa.load(file_audio, sr=None)    
    except FileNotFoundError as e:
        print(f"Audio file not found for number {number}: {e}")
        if number in labels_dict:
            del labels_dict[number]  # Remove the entry from the dictionary
        continue  # Skip to the next iteration if the audio file is not found
    

    # EXTRACT VISUAL
    try:
        with open(file_visual, "r") as f:
                # Skip first line (title)
                next(f)
                file_visual = f.readlines()
    except FileNotFoundError as e:
        print(f"Visual file not found for number {number}: {e}")
        continue  # Skip to the next iteration if the video file is not found

        # Convert list of strings into 2D numpy array
    visual_np = [np.fromstring(s, dtype=np.float32, sep=', ') for s in file_visual]
    visual = np.vstack(visual_np)

    # Load the transcript CSV file
    transcript_df = pd.read_csv(transcript_file, sep='\t')

    # Align audio and visual data
    # Step 1: Find the first start time in the transcript
    transcript_start_time = transcript_df['start_time'].iloc[0]
    # Get the ending timestamp from the transcript
    transcript_end_time = transcript_df['stop_time'].iloc[-1]  # Last timestamp in the transcript data

    # Step 2: Find the closest, greater or equal timestamp in the visual data
    # Extract the visual timestamps from the second column (I have double checked that it works steps 2,3,4)
    visual_timestamps = visual[:, 1]
    visual_start_idx = np.searchsorted(visual_timestamps, transcript_start_time, side='left')
    visual_end_idx = np.searchsorted(visual_timestamps, transcript_end_time, side='right') - 1


    # Check if visual_start_idx is valid and update the start time
    if visual_start_idx < len(visual_timestamps):
        visual_start_time = visual_timestamps[visual_start_idx]
        visual_end_time = visual_timestamps[visual_end_idx]
    else:
        raise ValueError("No valid start time found in visual data greater than transcript start time")
    
    print(f"Patient number: {number}")
    print(f"Visual start time: {visual_start_time}")
    print(f"Visual end time: {visual_end_time}")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Step 3: Trim the visual data to start from `visual_start_time` and end at `visual_end_time`
    visual = visual[visual_start_idx:visual_end_idx + 1]

    # Step 4: Calculate the corresponding start and end index for the audio data
    # Convert the visual start time to audio samples
    audio_start_idx = int(visual_start_time * sr)
    audio_end_idx = int(visual_end_time * sr)
    audio = audio[audio_start_idx:audio_end_idx + 1]

    # Now, both `visual` and `audio` are aligned to start and end at the same timestamps
    # print(f"Trimmed visual start time: {visual[0, 1]} seconds")
    # print(f"Trimmed audio start time: {audio_start_idx / sr} seconds")

    # print(f"Trimmed visual end time: {visual[-1, 1]} seconds")
    # print(f"Trimmed audio end time: {audio_end_idx / sr} seconds")


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

    # Save new audios

    # # Save the audio file
    # if(number==300):
    #     wavfile.write(final_audio_file, sr, final_audio.astype(np.float32))
    #     print(f"Audio saved to {final_audio_file}")

    # TODO: check it out
    preprocessed_audio = preprocess_audio(final_audio,sr)

    # Step 2: Calculate Mel-spectrogram for the final_audio
    n_fft = int(0.025 * sr)  # Window length: 25 ms
    hop_length = int(0.010 * sr)  # Hop length: 10 ms
    n_mels = 64  # Number of Mel bands

    stft = librosa.stft(preprocessed_audio, n_fft=n_fft, hop_length=hop_length, window='hann')
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

    # Load transcript start and stop times
    transcript_times = [(row['start_time'], row['stop_time']) for _, row in transcript_df.iterrows()]


    mel_segments = []
    reliable_mel_segments = []
    start_frame = 0
    cnt = 0
    frame_duration = hop_length / sr 
    while start_frame + frames_per_segment <= log_mel_spectrogram.shape[1]:
        end_frame = start_frame + frames_per_segment

        # Extract the Mel-spectrogram for the current window
        mel_segment = log_mel_spectrogram[:, start_frame:end_frame]
        mel_segments.append(mel_segment)
        cnt += 1
        # Move the window by the stride (0.1 seconds worth of frames)
        start_frame += frames_per_stride

    # Visualize the first segment
    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mel_segments[620], sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Log-Mel Spectrogram for 1st 3.5s Segment (Sliding Window)')
    # plt.tight_layout()
    # plt.show()

    # Step 5: Check number of segments and shapes
    # print(f"Number of segments: {len(mel_segments)}")
    # print(f"Shape of one Mel-spectrogram segment: {mel_segments[0].shape}")  # Should be (n_mels, frames_per_segment) / it is (64,350)

    # PROCESS VISUAL
    fps_au = 30  # 30 AU frames per second
    frames_per_segment_au = int(segment_duration * fps_au)  # 3.5 seconds worth of AU frames (105 frames)
    frames_per_stride_au = int(stride_duration * fps_au)    # 0.1 seconds worth of AU frames (3 frames)

    #  Step 1: Extract the AU features (ignoring metadata like frame, timestamp, confidence, success)

    # Extract regression and binary AUs
    r_indices = list(range(4, 18))  # Columns 4 to 17 correspond to AU*_r features
    c_indices = list(range(18, 24))  # Columns 18 to 23 correspond to AU*_c features
    # au_features_r = au_features.filter(regex='_r$')  # Select only regression outputs (continuous values)

    # Separate regression and binary features
    au_features_r = visual[:, r_indices]
    au_features_c = visual[:, c_indices]

    # Mask for reliability based on the original data
    reliability_mask = (visual[:, 3] == 1) 

    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize to [0, 1]
    # au_features_r_normalized = pd.DataFrame(scaler.fit_transform(au_features_r), columns=au_features_r.columns)
    au_features_r_normalized = scaler.fit_transform(au_features_r)

    # Combine both standardized regression and binary features into one dataset
    combined_au_features = np.concatenate([au_features_r_normalized, au_features_c], axis=1)

    # Print total number of frames and number of unreliable frames
    # (Checked and it matches)
    total_frames = len(reliability_mask)
    unreliable_frames = np.sum(~reliability_mask)  # Count frames where reliability is False

    # print(f"Total number of frames: {total_frames}")
    # print(f"Number of unreliable frames: {unreliable_frames}")
    # Filter out the unreliable rows using the reliability mask
    unreliable_rows = visual[~reliability_mask]

    # Sliding window segmentation of AUs
    au_segments = []
    reliable_au_segments = []    
    start_frame = 0
    mask_segments = []
    # Sliding window over the AU frames
    while start_frame + frames_per_segment_au <= len(combined_au_features):
        end_frame = start_frame + frames_per_segment_au

        au_segment = combined_au_features[start_frame:end_frame]  # Use numpy slicing  

        # Generate a mask segment for each visual segment
        mask_segment = reliability_mask[start_frame:end_frame]  # Take a slice of the reliability mask

        mask_segments.append(mask_segment)  # Append this mask segment
        au_segments.append(au_segment)
        # Move the window by the stride (0.1 seconds worth of frames)
        start_frame += frames_per_stride_au

    # print("Length of mask_segments:", len(mask_segments))
    # print("Length of au_segments:", len(au_segments))
    # print("Length of mel_segments:", len(mel_segments))

    if len(mel_segments) > len(au_segments):
        mel_segments.pop()  # Remove the last audio segment
    elif len(au_segments) > len(mel_segments):
        au_segments.pop()  # Remove the last visual segment


    # Append the segments to the lists
    aus.append(au_segments)
    log_mels.append(mel_segments)

    # Now filter based on mask_segments to keep only reliable segments
    for mel_segment, au_segment, mask_segment in zip(mel_segments, au_segments, mask_segments):
        if mask_segment.all():  # Only keep segments if all frames in the mask are reliable
            reliable_mel_segments.append(mel_segment)
            reliable_au_segments.append(au_segment)

    # Convert to numpy arrays if needed for storage
    # reliable_mel_segments = np.array(reliable_mel_segments, dtype=object)
    # reliable_au_segments = np.array(reliable_au_segments, dtype=object)

    log_mels_reliable.append(reliable_mel_segments)
    aus_reliable.append(reliable_au_segments)
    # Print counts to verify alignment
    print(f"Number of reliable audio segments: {len(reliable_mel_segments)}")
    print(f"Number of reliable visual segments: {len(reliable_au_segments)}")

    # Step 3: Check number of AU segments and shapes
    # print(f"Number of AU segments: {len(au_segments)}")
    # print(f"Shape of one AU segment: {au_segments[0].shape}")  # Should be (105, num_AUs) / it is (105,20)
    # print(f"Shape of one mask segment: {mask_segments[0].shape}")
    print(f"Patient {number}: Audio segments = {len(mel_segments)}, Visual segments = {len(au_segments)}")
# Save arrays in files
log_mels = np.array(log_mels, dtype=object)

# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/log_mels.npy', log_mels)
# # np.save('V:/staff-umbrella/EleniSalient/Preprocessing/log_mels.npy', log_mels)

aus = np.array(aus, dtype=object)

# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/aus.npy', aus)
# # np.save('V:/staff-umbrella/EleniSalient/Preprocessing/aus.npy', aus)

# Save arrays in files
log_mels_reliable = np.array(log_mels_reliable, dtype=object)

np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/log_mels_reliable_prep.npy', log_mels_reliable)
# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/log_mels_reliable.npy', log_mels_reliable)
# np.save('V:/staff-umbrella/EleniSalient/Preprocessing/log_mels.npy', log_mels)

aus_reliable = np.array(aus_reliable, dtype=object)

np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/aus_reliable_prep.npy', aus_reliable)
# # np.save('V:/staff-umbrella/EleniSalient/Preprocessing/aus.npy', aus)

# masks = np.array(masks, dtype=object)
# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/mask_segments.npy', masks)
# np.save('V:/staff-umbrella/EleniSalient/Preprocessing/mask_segments.npy', masks)

# print('Log-mels')
# print(log_mels.shape)
# print(len(log_mels[0]))
# print(len(log_mels[1]))

# print('Action Units')
# print(aus.shape)
# print(len(aus[0]))
# print(len(aus[1]))

# print('Log-mels_reliable')
# print(log_mels_reliable.shape)
# print(len(log_mels_reliable[0]))
# print(len(log_mels_reliable[1]))

# print('Action Units reliable')
# print(aus_reliable.shape)
# print(len(aus_reliable[0]))
# print(len(aus_reliable[1]))

# print('Masks')
# print(masks.shape)
# print(len(masks[0]))
# print(len(masks[1]))
