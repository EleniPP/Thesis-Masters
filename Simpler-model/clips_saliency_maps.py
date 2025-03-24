import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

def rolling_mean(values, window_size=5):
    """Compute the rolling mean for a 1D NumPy array."""
    # Pad the array at the start and end so the result has the same length
    half_w = (window_size - 1) // 2
    padded = np.pad(values, (half_w, half_w), mode='edge')
    smoothed = np.empty_like(values)
    
    for i in range(len(values)):
        smoothed[i] = padded[i : i + window_size].mean()
    
    return smoothed

def parse_interval(interval_str):
    """Parse something like '2-3 sec' into numeric floats (2.0, 3.0)."""
    interval_str = interval_str.replace(' sec', '').strip()
    start_str, end_str = interval_str.split('-')
    return float(start_str), float(end_str)


# Load the saved calibration results
calibration_results = torch.load('/tudelft.net/staff-umbrella/EleniSalient/calibration_results.pth')

# Access individual components
all_patient_numbers = torch.tensor(calibration_results["patient_ids"]) #the patient id of each segment
all_segment_orders = torch.tensor(calibration_results["segment_indices"])  #the index of each segment in the original order before the shuffling
predictions = torch.tensor(calibration_results["predictions"]) #the predictions of the model for each segment
true_labels = torch.tensor(calibration_results["true_labels"]) #true labels of each segment

# Sort all data by patient ID
sorted_patients, indices = torch.sort(all_patient_numbers)
sorted_segment_orders = all_segment_orders[indices]
sorted_true_labels = true_labels[indices]
sorted_predictions = predictions[indices]

# Group by patient
grouped_predictions = []
grouped_true_labels = []
# get unique values and how many times they appear
unique_values, counts = torch.unique(sorted_patients, return_counts=True)

# to keep track of where the groups start
start_idx = 0
# Iterate through each unique value
for value, count in zip(unique_values, counts):
    patients=sorted_patients[start_idx:start_idx + count] #each patients tensor contains the id of a patient num_of_segments times
    patient_segment_orders = sorted_segment_orders[start_idx:start_idx + count] #each patient_segment_orders contain the index of each segment which is the index from its original order before the shuffling
    labels = sorted_true_labels[start_idx:start_idx + count]
    preds = sorted_predictions[start_idx:start_idx + count]

    # Sort segments chronologically within the patient
    # Sort the segment orders to get the chronological order
    sorted_orders, sorted_order_indices = torch.sort(patient_segment_orders)  # Sort the orders and get indices
    # And now we also parallely sort the probabilities so they will also be in chronological order and that each probability will be alignes in its segment
    sorted_labels = labels[sorted_order_indices]
    sorted_preds = preds[sorted_order_indices]

    grouped_true_labels.append(sorted_labels)
    grouped_predictions.append(sorted_preds)

    # Update the starting index for the next group
    start_idx += count



per_patient_saliencies = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/per_patient_saliencies.npy', allow_pickle=True)
reliability_masks = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/reliability_masks.npy', allow_pickle=True)
unique_values = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/unique_values.npy', allow_pickle=True)
unique_values = torch.from_numpy(unique_values)

patient_id = 308
segment_id = 745
# patient_ids = [308, 321, 337, 338, 344, 365, 367, 380, 389, 440, 459, 483]
patient_ids= [308, 321, 337, 338, 344, 365, 367, 440, 459, 483,303, 323, 349, 401, 409, 411, 427, 445, 477, 490, 324, 379, 472, 478, 409, 352, 353, 405, 433, 448]
segment_ids = [745, 4754, 16174, 5117, 7635, 7522, 12626, 9279, 2298, 8720,3331, 1978, 3640, 3957, 1017, 5643, 4314, 5314, 2386, 4003,1118, 5277, 5990, 1095, 9666,2936, 2083, 12071, 1880, 5739]

# Define how many segments before and after to take for each clip
left_contexts = [25, 25, 25, 25, 25, 40, 40, 40, 10, 10, 10, 10, 10, 10, 40, 40, 40, 40, 25, 25, 25, 25, 40, 10, 10, 40, 40, 40, 25, 10, 25]
right_contexts = [25, 25, 25, 25, 25, 10, 10, 10, 40, 40, 40, 40, 40, 40, 10, 10, 10, 10, 25, 25, 25, 25, 10, 40, 40, 10, 10, 10, 25, 40, 25]


salient_time_arr = []
smoothed_saliencies_arr = []
saliency_values_arr = []

patient_id_to_group_index = {int(pid.item()): idx for idx, pid in enumerate(unique_values)}



# Arrays to store results
saliency_values_arr_TP = []
saliency_values_arr_TN = []
saliency_values_arr_FP = []
saliency_values_arr_FN = []
saliency_times_arr_TP = []
saliency_times_arr_TN = []
saliency_times_arr_FP = []
saliency_times_arr_FN = []
for i in range (len(patient_ids)):
    # Step 1: Find the patient index
    patient_index = torch.where(unique_values == patient_ids[i])[0].item()
    patient_index2 = patient_id_to_group_index[patient_ids[i]]

    # Step 2: Retrieve saliencies for that patient
    patient_saliencies = per_patient_saliencies[patient_index]

    # Step 3: Get the real segment indices (only reliable ones)
    original_indices = np.where(reliability_masks[patient_index] == 1)[0] 

    # Step 4: Find where segment 745 is in the original indices so I can take the index that it had when there were only reliable
    segment_index = np.where(original_indices == segment_ids[i])[0]

    if len(segment_index) == 0:
        raise ValueError(f"Segment {segment_ids[i]} is not reliable for patient {patient_ids[i]}.")
    segment_index = segment_index[0]

    # Retrieve prediction and true label
    prediction = grouped_predictions[patient_index][segment_index]
    true_label = grouped_true_labels[patient_index][segment_index]

    # Use the specific context window for this sample
    left = left_contexts[i]
    right = right_contexts[i]
    # print(f"Saliency for segment {segment_ids[i]} of patient {patient_ids[i]}: {saliency}")
    # Step 5: Get the 51 segments for plotting
    start_idx = max(segment_index - left, 0)  # Ensure it doesn't go out of bounds
    end_idx = min(segment_index + right, len(patient_saliencies))  # Ensure it doesn't exceed bounds

    # Extract saliencies for plotting
    saliency_values = patient_saliencies[start_idx:end_idx]
    print(type(saliency_values))
    print(type(saliency_values[0]))
    print(saliency_values.shape)

    # Suppose saliency_values is your 1D NumPy array of saliencies
    smoothed_saliencies = rolling_mean(saliency_values, window_size=5)

    # # Create x-axis labels (time relative to the salient segment)
    # time_axis = np.arange(start_idx - segment_index, end_idx - segment_index) * 0.1  # Convert index to seconds
    # **Create the time axis from 0 to 8.5 seconds**
    time_axis = np.linspace(0, 8.5, num=len(saliency_values))  

    # Find the correct X position for the vertical line
    salient_time = time_axis[segment_index - start_idx]  # Get the correct time
    smoothed_saliencies_arr.append(smoothed_saliencies)
    salient_time_arr.append(salient_time)
    saliency_values_arr.append(saliency_values)


    # Instead of appending each segment individually, create sub-arrays for each category:
    # window_TP_sal = []
    # window_TP_times = []
    # window_TN_sal = []
    # window_TN_times = []
    # window_FP_sal = []
    # window_FP_times = []
    # window_FN_sal = []
    # window_FN_times = []


    # for j in range(start_idx, end_idx):
    #     seg_saliency = patient_saliencies[j]  # Or smoothed_saliencies[j - start_idx] if desired
    #     seg_time = time_axis[j - start_idx]
    #     seg_prediction = grouped_predictions[patient_index][j]
    #     seg_true = grouped_true_labels[patient_index][j]
        
    #     if seg_prediction == 1 and seg_true == 1:
    #         window_TP_sal.append(seg_saliency)
    #         window_TP_times.append(seg_time)
    #     elif seg_prediction == 0 and seg_true == 0:
    #         window_TN_sal.append(seg_saliency)
    #         window_TN_times.append(seg_time)
    #     elif seg_prediction == 1 and seg_true == 0:
    #         window_FP_sal.append(seg_saliency)
    #         window_FP_times.append(seg_time)
    #     elif seg_prediction == 0 and seg_true == 1:
    #         window_FN_sal.append(seg_saliency)
    #         window_FN_times.append(seg_time)

    # # Append the entire window sub-array for this clip to the corresponding lists:
    # saliency_values_arr_TP.append(np.array(window_TP_sal))
    # saliency_times_arr_TP.append(np.array(window_TP_times))
    # saliency_values_arr_TN.append(np.array(window_TN_sal))
    # saliency_times_arr_TN.append(np.array(window_TN_times))
    # saliency_values_arr_FP.append(np.array(window_FP_sal))
    # saliency_times_arr_FP.append(np.array(window_FP_times))
    # saliency_values_arr_FN.append(np.array(window_FN_sal))
    # saliency_times_arr_FN.append(np.array(window_FN_times))

        # --- Create fixed-size arrays (with NaNs) for each category ---
    window_length = end_idx - start_idx
    window_TP_sal = np.full(window_length, np.nan, dtype=float)
    window_TN_sal = np.full(window_length, np.nan, dtype=float)
    window_FP_sal = np.full(window_length, np.nan, dtype=float)
    window_FN_sal = np.full(window_length, np.nan, dtype=float)

    window_TP_times = np.full(window_length, np.nan, dtype=float)
    window_TN_times = np.full(window_length, np.nan, dtype=float)
    window_FP_times = np.full(window_length, np.nan, dtype=float)
    window_FN_times = np.full(window_length, np.nan, dtype=float)

    for j in range(start_idx, end_idx):
        idx_in_window = j - start_idx
        seg_saliency = patient_saliencies[j]  # or smoothed_saliencies[idx_in_window]
        seg_time = time_axis[idx_in_window]
        seg_prediction = grouped_predictions[patient_index][j]
        seg_true = grouped_true_labels[patient_index][j]

        if seg_prediction == 1 and seg_true == 1:
            window_TP_sal[idx_in_window] = seg_saliency
            window_TP_times[idx_in_window] = seg_time
        elif seg_prediction == 0 and seg_true == 0:
            window_TN_sal[idx_in_window] = seg_saliency
            window_TN_times[idx_in_window] = seg_time
        elif seg_prediction == 1 and seg_true == 0:
            window_FP_sal[idx_in_window] = seg_saliency
            window_FP_times[idx_in_window] = seg_time
        elif seg_prediction == 0 and seg_true == 1:
            window_FN_sal[idx_in_window] = seg_saliency
        window_FN_times[idx_in_window] = seg_time

    # Append the entire window sub-array to the corresponding lists
    saliency_values_arr_TP.append(window_TP_sal)
    saliency_values_arr_TN.append(window_TN_sal)
    saliency_values_arr_FP.append(window_FP_sal)
    saliency_values_arr_FN.append(window_FN_sal)

    saliency_times_arr_TP.append(window_TP_times)
    saliency_times_arr_TN.append(window_TN_times)
    saliency_times_arr_FP.append(window_FP_times)
    saliency_times_arr_FN.append(window_FN_times)



smoothed_saliencies_arr = np.array(smoothed_saliencies_arr)
salient_time_arr = np.array(salient_time_arr)
saliency_values_arr = np.array(saliency_values_arr)
np.save(f'/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/smoothed_saliencies_arr.npy', smoothed_saliencies_arr)
np.save(f'/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/salient_time_arr.npy', salient_time_arr)
np.save(f'/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_values_arr.npy', saliency_values_arr)

# Convert and save arrays
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_values_arr_TP.npy', np.array(saliency_values_arr_TP, dtype=object))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_values_arr_TN.npy', np.array(saliency_values_arr_TN, dtype=object))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_values_arr_FP.npy', np.array(saliency_values_arr_FP, dtype=object))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_values_arr_FN.npy', np.array(saliency_values_arr_FN, dtype=object))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_times_arr_TP.npy', np.array(saliency_times_arr_TP, dtype=object))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_times_arr_TN.npy', np.array(saliency_times_arr_TN, dtype=object))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_times_arr_FP.npy', np.array(saliency_times_arr_FP, dtype=object))
np.save('/tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/saliency_times_arr_FN.npy', np.array(saliency_times_arr_FN, dtype=object))



print('DONE')
    # plt.figure(figsize=(10, 5))
    # plt.plot(time_axis, saliency_values, 'o-', label='Raw Saliency', alpha=0.3)
    # plt.plot(time_axis, smoothed_saliencies, 'r-', label='Smoothed Saliency')
    # plt.axvline(x=salient_time, color='maroon', linestyle='--', label='Salient Segment Start')
    # plt.title(f'Saliency Map for Patient {patient_ids[i]}, Segment {segment_ids[i]} (Smoothed)')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Saliency Score')
    # plt.legend()
    # plt.show()

    # Plot the saliency map
    # plt.figure(figsize=(10, 5))
    # plt.plot(time_axis, saliency_values, marker='o', color= 'darkslategrey',linestyle='-')
    # plt.axvline(x=salient_time, color='maroon', linestyle='--', label="Salient Segment Start")
    # plt.xlabel("Time (seconds)")
    # plt.ylabel("Saliency Score")
    # plt.xticks(np.arange(0, 9, 0.5))  # Set X-axis ticks every 0.5s
    # plt.title(f"Saliency Map for Patient {patient_ids[i]}, Segment {segment_ids[i]}")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'../../../tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/Clip_smooth_{patient_ids[i]}.png')
    # plt.close()
