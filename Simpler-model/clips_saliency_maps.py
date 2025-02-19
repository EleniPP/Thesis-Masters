import numpy as np
import torch
import matplotlib.pyplot as plt

per_patient_saliencies = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/per_patient_saliencies.npy', allow_pickle=True)
reliability_masks = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/reliability_masks.npy', allow_pickle=True)
unique_values = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/unique_values.npy', allow_pickle=True)
unique_values = torch.from_numpy(unique_values)

patient_id = 308
segment_id = 745
# patient_ids = [308, 321, 337, 338, 344, 365, 367, 380, 389, 440, 459, 483]
patient_ids = [308, 321, 337, 338]
# segment_ids = [745, 4754, 16174, 5117, 7635, 7522, 12626, 4115, 2218, 9279, 2298, 8720]
segment_ids = [745, 4754, 16174, 5117]

for i in range (len(patient_ids)):
    # Step 1: Find the patient index
    patient_index = torch.where(unique_values == patient_ids[i])[0].item()

    # Step 2: Retrieve saliencies for that patient
    patient_saliencies = per_patient_saliencies[patient_index]

    # Step 3: Get the real segment indices (only reliable ones)
    original_indices = np.where(reliability_masks[patient_index] == 1)[0] 

    # Step 4: Find where segment 745 is in the original indices so I can take the index that it had when there were only reliable
    segment_index = np.where(original_indices == segment_ids[i])[0]

    if len(segment_index) == 0:
        raise ValueError(f"Segment {segment_ids[i]} is not reliable for patient {patient_ids[i]}.")
    segment_index = segment_index[0]

    # # Step 5: Retrieve the saliency
    # saliency = patient_saliencies[segment_index]

    # print(f"Saliency for segment {segment_ids[i]} of patient {patient_ids[i]}: {saliency}")
    # Step 5: Get the 51 segments for plotting
    start_idx = max(segment_index - 25, 0)  # Ensure it doesn't go out of bounds
    end_idx = min(segment_index + 25, len(patient_saliencies))  # Ensure it doesn't exceed bounds

    # Extract saliencies for plotting
    saliency_values = patient_saliencies[start_idx:end_idx]

    # # Create x-axis labels (time relative to the salient segment)
    # time_axis = np.arange(start_idx - segment_index, end_idx - segment_index) * 0.1  # Convert index to seconds
    # **Create the time axis from 0 to 8.5 seconds**
    time_axis = np.linspace(0, 8.5, num=len(saliency_values))  

    # Find the correct X position for the vertical line
    salient_time = time_axis[segment_index - start_idx]  # Get the correct time

    # Plot the saliency map
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, saliency_values, marker='o', color= 'darkslategrey',linestyle='-')
    plt.axvline(x=salient_time, color='maroon', linestyle='--', label="Salient Segment Start")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Saliency Score")
    plt.xticks(np.arange(0, 9, 0.5))  # Set X-axis ticks every 0.5s
    plt.title(f"Saliency Map for Patient {patient_ids[i]}, Segment {segment_ids[i]}")
    plt.legend()
    plt.grid(True)
    # plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/Clip_{i}.png')
    plt.savefig(f'../../../tudelft.net/staff-umbrella/EleniSalient/Saliency_graphs_per_clip/Clip_{patient_ids[i]}.png')
    plt.close()
