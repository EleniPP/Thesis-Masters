import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def minutes_to_timestamp(minutes, seconds):
    """
    Convert minutes and seconds into a total timestamp in seconds.

    Args:
        minutes (int): The number of minutes.
        seconds (float): The number of seconds.

    Returns:
        float: The total timestamp in seconds.
    """
    return minutes * 60 + seconds

def parse_patient_time_file(file_path):
    patient_times = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):  # Process in blocks of 3 lines
            patient_number = int(lines[i].split(':')[1].strip())
            visual_start_time = float(lines[i + 1].split(':')[1].strip())
            visual_end_time = float(lines[i + 2].split(':')[1].strip())
            patient_times[patient_number] = visual_start_time
    return patient_times

def parse_saliency_file1(file_path):
    salient_features = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(':')
            patient_number = int(parts[0].split(',')[0].split()[1])   # Extract patient number
            indices = eval(parts[1].strip())  # Convert indices to list or int
            salient_features[patient_number] = indices if isinstance(indices, list) else [indices]
    return salient_features

def parse_saliency_file(file_path):
    salient_features = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(":")
            patient_number = int(parts[0].split()[1])  # Extract Patient number
            indices = eval(parts[1].strip())
            salient_features[patient_number] = indices if isinstance(indices, list) else [indices]
    return salient_features

def get_timestamp(segment_number, segment_duration, stride, start_timestamp):
    """
    Calculate the timestamp for a given segment in a sliding window setup.
    
    Args:
        segment_number: The segment index (starting from 1).
        segment_duration: Duration of each segment (in seconds).
        stride: Stride between consecutive segments (in seconds).
        start_timestamp: The timestamp at which segment 0 starts (in seconds).
    
    Returns:
        A formatted string representing the start and end timestamp of the segment.
    """
    # Calculate the start timestamp of the segment
    start_time_seconds = start_timestamp + (segment_number - 1) * stride

    # Calculate the end timestamp of the segment
    end_time_seconds = start_time_seconds + segment_duration

    # Convert start and end times to minutes and seconds
    start_minutes = int(start_time_seconds // 60)
    start_seconds = start_time_seconds % 60

    end_minutes = int(end_time_seconds // 60)
    end_seconds = end_time_seconds % 60

    return (f"Segment {segment_number} starts at {start_minutes} minutes and {start_seconds:.1f} seconds "
            f"and ends at {end_minutes} minutes and {end_seconds:.1f} seconds.") , (start_time_seconds,end_time_seconds)


# Function to extract salient rows and action unit columns
def extract_salient_segments(visual, timestamps, patient_number, au_columns):
    """
    Extract rows corresponding to salient segments and keep only AU columns.

    Args:
        visual (numpy.ndarray): Visual data array.
        timestamps (dict): Dictionary of patient IDs to lists of salient segment timestamps.
        patient_number (int): Patient ID to process.

    Returns:
        numpy.ndarray: Extracted rows with AU columns for the salient segments.
    """
    if patient_number not in timestamps:
        print(f"No salient segments found for patient {patient_number}.")
        return np.empty((0, len(au_columns)))  # Return empty array if no salient segments

    # Get the list of salient segments for the patient
    patient_segments = timestamps[patient_number]

    # Initialize a list to hold extracted rows
    extracted_rows = {}

    for start, end in patient_segments:
        # Filter rows where the timestamp falls within the start-end range
        segment_rows = visual[(visual[:, 1] >= start) & (visual[:, 1] <= end)]
        # extracted_rows.append(segment_rows[:, au_columns])
        extracted_rows[(start, end)] = segment_rows[:, au_columns]

    return extracted_rows

def plot_au_values(au_data, au_names, segment_name="Salient Segment"):
    """
    Plot intensity values of Action Units (AUs) over frames in a segment.

    Args:
        au_data (numpy.ndarray): The AU block with rows as frames and columns as AU values.
        au_names (list): List of AU names corresponding to the columns in au_data.
        segment_name (str): Name of the segment for the plot title.
    """
    frames = range(len(au_data))  # Frame indices
    plt.figure(figsize=(12, 6))
    for i, au_name in enumerate(au_names):
        plt.plot(frames, au_data[:, i], label=au_name)
    plt.xlabel("Frame")
    plt.ylabel("AU Intensity")
    plt.title(f"Action Unit Intensities in {segment_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"../../../tudelft.net/staff-umbrella/EleniSalient/Results/AUs_visualization_{segment_name}.png")
    plt.close()


# Function to plot AU clusters
def plot_au_clusters(extracted_au_data, au_names, segment_name):
    for cluster_name, indices in au_clusters.items():
        cluster_data = extracted_au_data[:, indices]  # Select columns for this cluster
        cluster_names = [au_names[i] for i in indices]  # Get AU names for the cluster
        plot_au_values(cluster_data, cluster_names, segment_name=f"{segment_name} - {cluster_name}")

def get_previous_segment(visual, current_segment_start_timestamp, segment_size=3.5, segment_stride=3, au_columns=None):
    """
    Get the data for the previous segment.

    Parameters:
        visual (np.array): The entire visual data with rows for each frame.
        current_segment_start_index (int): The starting index of the current salient segment.
        frames_per_segment (int): The number of frames per segment (default: 105).
        frame_stride (int): The stride of the sliding window (default: 3).
        au_columns (list or None): Indices of AU columns to extract (default: None for all columns).

    Returns:
        np.array: The AU data for the previous segment, or an empty array if no previous segment exists.
    """
    # Calculate the starting index of the previous segment
    previous_start_time = current_segment_start_timestamp - segment_stride
    previous_end_time = previous_start_time + segment_size

    # Ensure the previous segment is within bounds
    if previous_start_time < 0:
        print("No previous segment exists.")
        return np.empty((0, visual.shape[1] if au_columns is None else len(au_columns)))
    
    # Filter rows based on timestamps
    previous_segment = visual[(visual[:, 1] >= previous_start_time) & (visual[:, 1] < previous_end_time)]

    previous_segment = previous_segment[:, au_columns]

    return previous_segment

# Function to compute transition rate
def compute_transition_rate(frames_binary):
    transitions = np.abs(np.diff(frames_binary, axis=0))  # Compute frame-to-frame transitions
    transition_rate = np.sum(transitions, axis=0) / (len(frames_binary) - 1)  # Normalize by frame count
    return transition_rate

def measure_au_changes(segment_array, previous_segment, au_columns):
    """
    Measure the change in AU values between the current and previous segment.

    Parameters:
        segment_array (np.array): The current segment's AU values.
        previous_segment (np.array): The previous segment's AU values.
        au_columns (list): Indices of AU columns.

    Returns:
        dict: A dictionary containing changes for added and removed frames.
    """

    # Isolate the last 3 frames (last 0.1 seconds) of the segment
    added_frames = segment_array[-3:]
    remaining_frames = segment_array[:-3]

    binary_added_frames = added_frames[:, -6:]  # Last 6 columns
    binary_remaining_frames = remaining_frames[:, -6:]  # Last 6 columns
    binary_previous_segment = previous_segment[:, -6:]  # Last 6 columns

    previous_segment_avg = previous_segment.mean(axis=0, keepdims=True)  # Average over the previous segment

    # Compare last 3 frames with the mean of the rest of the segment
    frame_differences = added_frames - remaining_frames.mean(axis=0, keepdims=True)

    # Aggregate AU changes across the last 3 frames
    au_change = np.mean(frame_differences, axis=0)  # Chnage due to the added frames (compared to rest of the segment)
    added_change = np.mean(added_frames - previous_segment_avg, axis=0) # Change due to the added frames (compared to previous segment)
# --------------------------------------------------------------------------------------------
    # # Determine the frames for comparison
    # added_frames = segment_array[-3:]  # Last 0.1 seconds (3 frames) of the current segment
    # removed_frames = previous_segment[:3]  # First 0.1 seconds (3 frames) of the previous segment

    # # Calculate the averages
    # previous_segment_avg = previous_segment.mean(axis=0)  # Average over the previous segment
    # added_avg = added_frames.mean(axis=0)  # Average over the added frames
    # removed_avg = removed_frames.mean(axis=0)  # Average over the removed frames

    # # Compute the changes
    # added_change = added_avg - previous_segment_avg  # Change due to the added frames
    # removed_change = removed_avg - previous_segment_avg  # Change due to the removed frames


    # Return the changes
    # return {
    #     "added_change": added_change,
    #     "removed_change": au_change,
    # }
    return {"added_change": added_change}


def visualize_au_change(patient_id ,visual, salient_segments, au_columns,au_names, segment_name=""):
    """
    Visualize the change in AU values from the previous segment to salient segments.
    
    Parameters:
        visual_data (np.array): The visual data with all columns (frame, timestamp, AUs, etc.).
        salient_indices (list): Indices of salient segments.
        au_columns (list): Indices of columns corresponding to AU values.
        segment_name (str): Name of the segment for labeling the plot.
    """
    # Create a folder for the patient
    base_path = "../../../tudelft.net/staff-umbrella/EleniSalient/Results"
    patient_folder = os.path.join(base_path, f"Patient_{patient_id}")
    os.makedirs(patient_folder, exist_ok=True)

    # Iterate over each salient segment's array for this patient
    for (start, end), segment_array in salient_segments.items():
         # Use the `get_previous_segment` function directly with timestamps
        previous_segment = get_previous_segment(
            visual=visual,
            current_segment_start_timestamp=start,  # Pass the timestamp directly
            au_columns=au_columns
        )

        if previous_segment.size == 0:
            print(f"Skipping segment {start}-{end}: No previous segment available.")
            continue

        # Calculate the change in AU values
         # Measure AU changes
        changes = measure_au_changes(segment_array, previous_segment, au_columns)

        # Plot the changes
        for change_type, change_values in changes.items():
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(au_columns)), change_values, tick_label=au_names)
            plt.xlabel("Action Units")
            plt.ylabel("Change in Intensity")
            plt.title(f"{change_type.capitalize()} - Segment ({start:.1f} - {end:.1f} seconds)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save the plot in the patient's folder
            file_path = os.path.join(
                patient_folder,
                f"AU_change_{segment_name}_{start:.1f}_{end:.1f}_{change_type}.png"
            )
            # Save the plot
            plt.savefig(file_path)
            plt.close()


def visualize_au_progress(patient_id ,visual, salient_segments, au_indices, au_names,au_cluster, segment_name=""):

    # Create a folder for the patient
    base_path = "../../../tudelft.net/staff-umbrella/EleniSalient/Results"
    patient_folder = os.path.join(base_path, f"Patient_{patient_id}")
    os.makedirs(patient_folder, exist_ok=True)

    # Iterate over each salient segment's array for this patient
    for (start, end), segment_array in salient_segments.items():
         # Use the `get_previous_segment` function directly with timestamps
        previous_segment = get_previous_segment(
            visual=visual,
            current_segment_start_timestamp=start,  # Pass the timestamp directly
            au_columns=au_columns
        )

        if previous_segment.size == 0:
            print(f"Skipping segment {start}-{end}: No previous segment available.")
            continue

    # segment_array, previous_segment,au_names,au_index
        segment_array = segment_array[:, au_indices]
        previous_segment = previous_segment[:, au_indices]
        added_frames = segment_array[-3:]
        # Combine AU values: Take all frames from segment_s_minus_1 and append the last frame from segment_s
        combined_au_values = np.concatenate([previous_segment, added_frames],axis=0)

            # Create time axis for 3.6 seconds (105 frames + 1 extra frame)
        time_axis = np.linspace(0, 3.6, len(combined_au_values))
            # Get the name of the AU based on the index
        
        # Plot the combined AU progress
        plt.figure(figsize=(10, 6))
        for i, au_index in enumerate(au_indices):
            plt.plot(time_axis, combined_au_values[:, i], label=au_names[au_index], marker='o', linestyle='-', alpha=0.8)
        # plt.plot(time_axis, combined_au_values, label=f"{au_name}", marker='o', linestyle='-', alpha=0.8)

        # Add labels, title, and legend
        plt.xlabel("Time (seconds)")
        plt.ylabel("AU Intensity")
        plt.title(f"Progress of Selected AU Over Combined Duration")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        # Save the plot in the patient's folder
        file_path = os.path.join(
            patient_folder,
            f"AU_progress_{segment_name}_{start:.1f}_{end:.1f}_{au_cluster}.png"
        )
        # Save the plot
        plt.savefig(file_path)
        plt.close()
        print(1/0)

if __name__ == "__main__":
    # Example usage
    segment_duration = 3.5  # Each segment spans 3.5 seconds
    stride = 0.1  # The stride between segments is 0.1 seconds

    au_columns = [
        4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    ]  # Indices for AU*_r and AU*_c columns

    au_names = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU09_r", "AU10_r",
            "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU25_r", "AU26_r",
            "AU04_c", "AU12_c", "AU15_c", "AU23_c", "AU28_c", "AU45_c"]


    # Define AU clusters
    au_clusters = {
        "Brow Region": [0, 1, 2],  # AU01_r, AU02_r, AU04_r
        "Eye Region": [3, 19],     # AU05_r, AU45_c
        "Cheeks": [4, 7],          # AU06_r, AU12_r
        "Mouth Region": [5, 8, 9, 10, 11],  # AU10_r, AU14_r, AU15_r, AU17_r, AU20_r
        "Jaw Region": [12, 13],    # AU25_r, AU26_r
        "Binary AUs": [14, 15, 16, 17, 18, 19]  # AU*_c columns
    }

    # Indices for regression (_r) and classification (_c) AUs
    au_r_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # AU*_r
    au_c_indices = [14, 15, 16, 17, 18, 19]  # AU*_c

    # Names for each group
    au_r_names = au_names[:14]  # First 14 are regression AUs
    au_c_names = au_names[14:]  # Last 6 are classification AUs

    # patient_times = parse_patient_time_file("C:/Users/eleni/Thesis-Masters/Simpler-model/timestamps_per_patient.txt")
    patient_times = parse_patient_time_file("./timestamps_per_patient.txt")		

    # salient_segments = parse_saliency_file("C:/Users/eleni/Thesis-Masters/Simpler-model/salient_segments_per_patient.txt")
    salient_segments = parse_saliency_file("./salient_indices.txt")

    # Map the salient features to the correct patient ID
    # mapped_salient_features = {}
    # start_time_keys = list(patient_times.keys())  # Get the patient IDs in order

    # for salient_patient_index, salient_indices in salient_segments.items():
    #     actual_patient_id = start_time_keys[salient_patient_index]  # Get the corresponding patient ID
    #     mapped_salient_features[actual_patient_id] = salient_indices  # Map salient indices to the correct patient ID

    # # Map salient features to timestamps
    # timestamps_for_audio = {}
    # timestamps_for_visual = {}
    # for patient_id, salient_indices in mapped_salient_features.items():
    #     if patient_id in patient_times:  # Ensure patient exists in start_times
    #         start_time = patient_times[patient_id]
    #         # Map each salient index to a timestamp
    #         timestamps = [get_timestamp(index, segment_duration, stride, start_time)[0] for index in salient_indices]
    #         times = [get_timestamp(index, segment_duration, stride, start_time)[1] for index in salient_indices]
    #         timestamps_for_audio[patient_id] = timestamps
            # timestamps_for_visual[patient_id] = times   

    participants_segments = {
        "TP": {
            "Participant 1": [(319, 1240), (353, 5942), (388, 1939), (423, 5662)],
            "Participant 2": [(423, 3967), (454, 2528), (461, 6896), (354, 3195)],
            "Participant 3": [(442, 3575), (362, 2739), (367, 9685), (321, 1382)],
            "Participant 4": [(389, 4888), (380, 401), (377, 4374), (434, 5082)],
            "Participant 5": [(338, 1655), (389, 4888), (376, 918), (359, 1619)],
            "Participant 6": [(386, 7510), (337, 3797), (325, 2335), (330, 5699)],
        },
        "FP": {
            "Participant 1": [(409, 3651), (470, 3378), (385, 2425), (379, 552)],
            "Participant 2": [(302, 4648), (431, 3268), (387, 3777), (306, 415)],
            "Participant 3": [(385, 4762), (472, 9630), (471, 1998), (371, 4985)],
            "Participant 4": [(452, 349), (383, 8838), (306, 3335), (431, 3268)],
            "Participant 5": [(387, 3777), (318, 931), (392, 1221), (447, 5012)],
            "Participant 6": [(302, 4648), (387, 3777), (483, 4877), (431, 3268)],
        },
        "TN": {
            "Participant 1": [(475, 5678), (489, 1480), (452, 48), (317, 1801)],
            "Participant 2": [(315, 2570), (453, 932), (471, 3683), (412, 2031)],
            "Participant 3": [(328, 6098), (371, 6273), (307, 5985), (436, 7128)],
            "Participant 4": [(430, 2299), (475, 5678), (373, 2471), (360, 2264)],
            "Participant 5": [(329, 2082), (305, 12201), (371, 6273), (343, 1656)],
            "Participant 6": [(387, 3777), (431, 3268), (407, 4456), (478, 9704)],
        },
        "FN": {
            "Participant 1": [(449, 9482), (389, 6178), (330, 2512), (330, 0)],
            "Participant 2": [(434, 7259), (344, 862), (423, 7907), (325, 4861)],
            "Participant 3": [(339, 1390), (356, 1323), (441, 7247), (389, 6178)],
            "Participant 4": [(389, 286), (335, 7666), (449, 573), (339, 1390)],
            "Participant 5": [(419, 2356), (344, 862), (308, 3944), (449, 9482)],
            "Participant 6": [(431, 3268), (387, 3777), (441, 1434), (389, 6178)],
        },
    }


    # Translate segment indices to timestamps
    translated_segments = {}

    for category, participants in participants_segments.items():
        translated_segments[category] = {}
        for participant, segments in participants.items():
            translated_segments[category][participant] = []
            for patient_id, segment_index in segments:
                if patient_id in patient_times:
                    start_timestamp = patient_times[patient_id]
                    timestamp = get_timestamp(segment_index, segment_duration, stride, start_timestamp)[1]
                    translated_segments[category][participant].append((patient_id, segment_index, timestamp))

    print(translated_segments)
    # print(timestamps_for_visual)

    # base_path = "/tudelft.net/staff-umbrella/EleniSalient/"
    # patient = "_P/"
    # visual_extension = "_CLNF_AUs.txt"

    # # numbers = list(range(300, 491))
    # numbers = [319]

    # for number in numbers:
    #     file_visual = f"{base_path}{number}{patient}{number}{visual_extension}"

    #     # EXTRACT VISUAL
    #     try:
    #         with open(file_visual, "r") as f:
    #                 # Skip first line (title)
    #                 next(f)
    #                 file_visual = f.readlines()
    #     except FileNotFoundError as e:
    #         print(f"Visual file not found for number {number}: {e}")
    #         continue  # Skip to the next iteration if the video file is not found

    #     visual_np = [np.fromstring(s, dtype=np.float32, sep=', ') for s in file_visual]
    #     visual = np.vstack(visual_np)

    #     # Assuming `visual` is already loaded as a numpy array
    #     extracted_au_data = extract_salient_segments(visual, timestamps_for_visual, number,au_columns)

    #     # # Separate the data for plotting
    #     # au_r_data = extracted_au_data[:, au_r_indices]
    #     # au_c_data = extracted_au_data[:, au_c_indices]
        
    #     # Plot each cluster
    #     # plot_au_clusters(extracted_au_data, au_names, segment_name=f"Patient {number}")

    #     # Convert dictionary items to a list and access the first item
    #     first_region, first_indices = list(au_clusters.items())[0]

    #     # visualize_au_change(patient_id=number, visual=visual, salient_segments=extracted_au_data, au_columns=au_columns, au_names=au_names, segment_name=number)
    #     visualize_au_progress(patient_id=number, visual=visual, salient_segments=extracted_au_data, au_indices=first_indices, au_names=au_names,au_cluster=first_region, segment_name=number)

    
