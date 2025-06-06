import torch
import matplotlib.pyplot as plt
import numpy as np
import collections


def entropy(probabilities):
    return -(probabilities * torch.log(probabilities)).sum(dim=-1)

def plot_saliency_map(saliency_values, file_name,  title='Saliency Map', save=True):
    step = 10  # Plot every n-th segment
    sampled_saliency = saliency_values[::step]  # Downsample saliency values
    sampled_indices = range(0, len(saliency_values), step)  # Downsample indices

    plt.figure(figsize=(10, 5))
    # plt.plot(saliency_values, label='Saliency')
    plt.plot(sampled_indices, sampled_saliency, label=f"Sampled (every {step})", color='olive')
    plt.xlabel('Segment')
    plt.ylabel('Normalized Saliency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(f'/tudelft.net/staff-umbrella/EleniSalient/Results/Saliency_maps/{file_name}')
        plt.close()
    else:
        plt.show()


def normalize_saliency(saliency_values):
    max_val = torch.max(saliency_values)
    min_val = torch.min(saliency_values)
    normalized_saliency = (saliency_values - min_val) / (max_val - min_val)
    return normalized_saliency


# Load the saved calibration results
calibration_results = torch.load('/tudelft.net/staff-umbrella/EleniSalient/calibration_results.pth')

# Access individual components
all_patient_numbers = torch.tensor(calibration_results["patient_ids"]) #the patient id of each segment
all_segment_orders = torch.tensor(calibration_results["segment_indices"])  #the index of each segment in the original order before the shuffling
predictions = torch.tensor(calibration_results["predictions"]) #the predictions of the model for each segment
true_labels = torch.tensor(calibration_results["true_labels"]) #true labels of each segment
probability_raw = torch.tensor(calibration_results["calibrated_probs"]) #the probability distribution of the model for each segment
probability_distributions = torch.tensor(calibration_results["calibrated_probs_platt"]) #the probability distribution of the model for each segment

reliability_masks = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/reliability_masks.npy', allow_pickle=True)

sorted_patient_ids = sorted(set(calibration_results["patient_ids"]))

# for patient_id, mask in zip(sorted_patient_ids, reliability_masks):
#     total_segments = len(mask)  # Total number of segments for this patient
#     num_reliable_segments = np.sum(mask)  # Number of reliable segments (where mask == 1)
#     # Double checked with the preprocessing and it is correct!! YEYYYY
#     print(f"Patient {patient_id}: {total_segments} total segments, {num_reliable_segments} reliable segments")

# Count occurrences of each patient ID
segment_counts = collections.Counter(all_patient_numbers.tolist())


print('Loaded tensors shapes:')
print(all_patient_numbers.shape)
print(all_segment_orders.shape)
print(predictions.shape)
print(true_labels.shape)
print(probability_distributions.shape)


# now this one has shuffled patient numbers. Each patient number appears in the torch as many times as the segments the patient video has. Each patient number
# is in the position of the list where the respective segment is on the probability distribution list of the segment.
torch.set_printoptions(threshold=torch.inf)

# So what i want is for each patient, to have a probability distribution so i can calculate the jacobian and the saliency 
# So first we have to align again the probability distribution torch with the all patient numbers torch after we put in ascending order the all patient numbers torch.

# Step 1: Sort everything by patient IDs
sorted_patients, indices = torch.sort(all_patient_numbers)
# Then we use the indices to sort the probability distributions in the same order as the patients were sorted so that each segment belongs to the right patient
sorted_probability_distributions = probability_distributions[indices]
sorted_segment_orders = all_segment_orders[indices]
sorted_true_labels = true_labels[indices]  # Align true labels with sorted segments
sorted_predictions = predictions[indices]  # Align predictions with sorted segments

# Step 2: Group by patient
# Now create a tensor of probability distributions for each patient. So i need to group sorted patients by patient.
grouped_patients = []
grouped_probabilities = []
grouped_true_labels = []
grouped_predictions = []

# get unique values and how many times they appear
unique_values, counts = torch.unique(sorted_patients, return_counts=True)
mapped_segment_orders = []  # Stores mapped segment indices per patient

# Iterate over each patient and its corresponding reliability mask
for mask in reliability_masks:  
    # Get the indices where the segment is reliable
    true_indices = np.where(mask == 1)[0]  # Get original indices of reliable segments
    
    # Append the correct indices
    mapped_segment_orders.append(torch.tensor(true_indices))  # Convert to tensor if needed

print(unique_values)
# unique_values = unique_values.numpy()
# print(unique_values)
# np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/unique_values.npy', unique_values)
# print(1/0)


# to keep track of where the groups start
start_idx = 0
# Iterate through each unique value
for value, count in zip(unique_values, counts):
    patients=sorted_patients[start_idx:start_idx + count] #each patients tensor contains the id of a patient num_of_segments times
    patient_segment_orders = sorted_segment_orders[start_idx:start_idx + count] #each patient_segment_orders contain the index of each segment which is the index from its original order before the shuffling
    probabilities = sorted_probability_distributions[start_idx:start_idx + count] #each probabilities tensor contains the probability distribution of the segment in the aligned position with the above.
    labels = sorted_true_labels[start_idx:start_idx + count]
    preds = sorted_predictions[start_idx:start_idx + count]

    # Sort segments chronologically within the patient
    # Sort the segment orders to get the chronological order
    sorted_orders, sorted_order_indices = torch.sort(patient_segment_orders)  # Sort the orders and get indices
    # And now we also parallely sort the probabilities so they will also be in chronological order and that each probability will be alignes in its segment
    sorted_probabilities = probabilities[sorted_order_indices]
    sorted_labels = labels[sorted_order_indices]
    sorted_preds = preds[sorted_order_indices]

    # Slice the tensors based on the count of each unique value
    grouped_patients.append(sorted_patients[start_idx:start_idx + count]) #slice starts at start_idx and ends at start_idx+count
    grouped_probabilities.append(sorted_probabilities)
    grouped_true_labels.append(sorted_labels)
    grouped_predictions.append(sorted_preds)

    # Update the starting index for the next group
    start_idx += count


step = 100  # Inspect every 3rd segment
global_threshold = 0.76  # Set to the global 99th percentile
threshold = 0.8  # Set to the local threshold
all_normalized_saliencies = []
# Open two files for writing salient indices and correctness
# with open("salient_indices.txt", "w") as indices_file, open("salient_correctness.txt", "w") as correctness_file:
# Open files for writing TP, FP, TN, and FN segments

# Comment out because we are calculating the lowest salient segments
with open("true_positives.txt", "w") as tp_file, \
     open("false_positives.txt", "w") as fp_file, \
     open("true_negatives.txt", "w") as tn_file, \
     open("false_negatives.txt", "w") as fn_file:
    
    all_entropies = []
    per_patient_saliencies = []
    for i, probability_distribution in enumerate(grouped_probabilities):
        # Extract the actual patient ID
        actual_patient_id = unique_values[i].item()

        #     # Skip if not the target patient
        # if actual_patient_id != 308:
        #     continue 
        # Compute the entropy for each probability distribution (each row in the tensor)
        entropies = entropy(probability_distribution)

        entropies_np = entropies.numpy()
        # Store entropies for plotting
        all_entropies.append(entropies_np)

        # Compute the gradient of entropy across selected segments
        jacobian = np.gradient(entropies_np)

        # Since this is a 1D problem, J'(x)J(x) is 1x1, therefore determinant
        # is the square of gradient itself
        patient_saliency = np.square(jacobian)

        # Normalize saliency
        normalized_saliency = normalize_saliency(torch.tensor(patient_saliency))
        all_normalized_saliencies.extend(normalized_saliency.numpy())  # Collect into the list
        per_patient_saliencies.append(normalized_saliency.numpy()) # Collect per patient

        # ---- **Map the segment indices correctly** ----
        # Take into consideration the segments that were removed in preprocessing_presegment because they are unreliable and map the indices of the segments correctly
        # Find the correct index of the patient in `unique_values` (which contains all patient IDs in sorted order) this is because reliability masks dont have the paatients ids integrated
        # so you cant do reliability_mask[actual_patient_id] because its out of bounds (there is no reliability_mask[300]) 

        original_indices = np.where(reliability_masks[i] == 1)[0]  # Get real segment indices

        if len(original_indices) != len(normalized_saliency):
            raise ValueError(f"Mismatch for patient {actual_patient_id}: {len(original_indices)} real vs {len(normalized_saliency)} saliencies")

        # ---- **Top 5 salient segments (CORRECT MAPPING)** ----
        top_k = 5  
        salient_indices = torch.topk(normalized_saliency, top_k).indices.tolist()  # Indices of top 5 salient segments
        real_salient_indices = [original_indices[idx] for idx in salient_indices]  # Map to real dataset indices
        final_saliency = normalized_saliency[salient_indices]  # Get corresponding saliency values

        # Separate the indices into TP, FP, TN, FN
        tp_segments = []
        fp_segments = []
        tn_segments = []
        fn_segments = []

        # # for idx in salient_indices:
        # for idx, real_idx in zip(salient_indices, real_salient_indices):
        #     # To retrieve entries from grouped_true_labels and grouped_predictions we dont use the real_idx because they only contain the reliable segments
        #     true_label = grouped_true_labels[i][idx].item()  # True label of the salient segment
        #     prediction = grouped_predictions[i][idx].item()  # Model's prediction for the salient segment

        #     # The real index is only used for the mapping of the salient segments to their real index so later I can calculate their true timestamps
        #     if true_label == 1 and prediction == 1:
        #         tp_segments.append(real_idx)  # True Positive
        #     elif true_label == 0 and prediction == 1:
        #         fp_segments.append(real_idx)  # False Positive
        #     elif true_label == 0 and prediction == 0:
        #         tn_segments.append(real_idx)  # True Negative
        #     elif true_label == 1 and prediction == 0:
        #         fn_segments.append(real_idx)  # False Negative

        # # Write results to files
        # tp_file.write(f"Patient {actual_patient_id}: {[int(seg) for seg in tp_segments]}\n")
        # fp_file.write(f"Patient {actual_patient_id}: {[int(seg) for seg in fp_segments]}\n")
        # tn_file.write(f"Patient {actual_patient_id}: {[int(seg) for seg in tn_segments]}\n")
        # fn_file.write(f"Patient {actual_patient_id}: {[int(seg) for seg in fn_segments]}\n")
per_patient_saliencies = np.array(per_patient_saliencies, dtype=object)
np.save('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/per_patient_saliencies.npy', per_patient_saliencies)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Calculate the lowest salient segments
# selected_patients = {325, 332, 347, 352, 359, 372, 389, 402, 414, 441, 453, 461}

# with open("tp_low.txt", "w") as tp_low_file:
#     all_entropies = []
#     per_patient_saliencies = []
    
#     for i, probability_distribution in enumerate(grouped_probabilities):
#         # Extract the actual patient ID
#         actual_patient_id = unique_values[i].item()

#         # Only process the selected patients
#         if actual_patient_id not in selected_patients:
#             continue  

#         # Compute the entropy for each probability distribution (each row in the tensor)
#         entropies = entropy(probability_distribution)
#         entropies_np = entropies.numpy()

#         # Store entropies for plotting
#         all_entropies.append(entropies_np)

#         # Compute the gradient of entropy across selected segments
#         jacobian = np.gradient(entropies_np)
#         patient_saliency = np.square(jacobian)  # Compute saliency as the squared gradient

#         # Normalize saliency
#         normalized_saliency = normalize_saliency(torch.tensor(patient_saliency))
#         all_normalized_saliencies.extend(normalized_saliency.numpy())  # Collect into the list
#         per_patient_saliencies.append(normalized_saliency.numpy())  # Collect per patient

#         # ---- **Map the segment indices correctly** ----
#         original_indices = np.where(reliability_masks[i] == 1)[0]  # Get real segment indices

#         if len(original_indices) != len(normalized_saliency):
#             raise ValueError(f"Mismatch for patient {actual_patient_id}: {len(original_indices)} real vs {len(normalized_saliency)} saliencies")

#         # ---- **Find the lowest 5 salient segments instead of highest** ----
#         top_k = 5  
#         salient_indices = torch.topk(normalized_saliency, top_k, largest=False).indices.tolist()  # Indices of lowest 5 salient segments
#         real_salient_indices = [original_indices[idx] for idx in salient_indices]  # Map to real dataset indices
#         final_saliency = normalized_saliency[salient_indices]  # Get corresponding saliency values

#         # Separate the indices into TP (since we are saving only TP)
#         tp_segments = []

#         for idx, real_idx in zip(salient_indices, real_salient_indices):
#             true_label = grouped_true_labels[i][idx].item()  # True label of the salient segment
#             prediction = grouped_predictions[i][idx].item()  # Model's prediction for the salient segment

#             if true_label == 1 and prediction == 1:
#                 tp_segments.append(real_idx)  # True Positive

#         # Write results to the `tp_low.txt` file
#         tp_low_file.write(f"Patient {actual_patient_id}: {[int(seg) for seg in tp_segments]}\n")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # total_segments = len(grouped_probabilities[i])  # Number of segments for this patient

    # Save results to files
    #     indices_file.write(f"Patient {actual_patient_id}: {salient_indices}\n")
    #     correctness_file.write(f"Patient {actual_patient_id}: {salient_correctness}\n")
        # print(f"Patient {grouped_patients[i][0]}, Final Selected Saliencies: {salient_indices}")
        # print(f"Patient {i}, Salient Classification Correctness: {salient_correctness}")
#  Plot entropy trends for each patient
# plt.figure(figsize=(10, 6))
# for i, entropies in enumerate(all_entropies[::15]):
#     plt.plot(entropies, label=f'Patient {i}')

# plt.title("Entropy Trends Across Segments for All Patients")
# plt.xlabel("Segment Index")
# plt.ylabel("Entropy")
# plt.legend(loc="upper right", fontsize='small')
# plt.grid()
# plt.savefig('/tudelft.net/staff-umbrella/EleniSalient/Results/Saliency_maps/entropy_trends.png')
    # Previous version with threshold 0.8
    # high_saliency_indices = torch.nonzero(normalized_saliency > threshold).view(-1).tolist()
    # print(f"Patient {i}, High Saliency Indices: {len(high_saliency_indices)}")


    # if i == 0 or i == 19:
    #     plot_saliency_map(normalized_saliency.numpy(), f'patient_{grouped_patients[i][0]}_saliency.png', title='Saliency Map', save=True)
    #     plt.bar(range(len(normalized_saliency)), normalized_saliency, color=["green" if predictions[i] == true_labels[i] else "red" for i in range(len(normalized_saliency))])
    #     plt.xlabel("Segment Index")
    #     plt.ylabel("Normalized Saliency")
    #     plt.title(f"Saliency Map - Patient {i}")
    #     plt.savefig(f'/tudelft.net/staff-umbrella/EleniSalient/Results/Saliency_maps/patient_{grouped_patients[i][0]}_saliency.png')


# # Convert to numpy array for easier analysis
# all_normalized_saliencies = np.array(all_normalized_saliencies)

# # Compute global statistics
# mean_saliency = np.mean(all_normalized_saliencies)
# median_saliency = np.median(all_normalized_saliencies)
# percentiles = np.percentile(all_normalized_saliencies, [25, 50, 75, 90, 95, 98, 99]) #Percentiles: [0.00163436. 0.02142086, 0.11816858, 0.31782514, 0.48015048, 0.66558971, 0.76908864]

# print(f"Mean Saliency: {mean_saliency}")
# print(f"Median Saliency: {median_saliency}")
# print(f"Percentiles: {percentiles}")

# # Plot histogram
# plt.hist(all_normalized_saliencies, bins=50, edgecolor='k', alpha=0.7)
# plt.title("Global Saliency Distribution")
# plt.xlabel("Saliency Value")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.savefig('/tudelft.net/staff-umbrella/EleniSalient/Results/Saliency_maps/global_saliency_distribution.png')



# Collect median saliencies for sorting
patient_medians = [
    np.median(normalize_saliency(torch.tensor(patient_saliency)).numpy())
    for patient_saliency in grouped_probabilities
]

# Sort patients by median saliency
sorted_indices = np.argsort(patient_medians)
sorted_patient_ids = [unique_values[i].item() for i in sorted_indices]
sorted_saliencies = [per_patient_saliencies[i] for i in sorted_indices]

# Create box plot
plt.figure(figsize=(20, 8))
plt.boxplot(sorted_saliencies, showfliers=False, notch=True)
plt.xticks(
    ticks=range(1, len(sorted_patient_ids) + 1),
    labels=sorted_patient_ids,
    rotation=90,
)
plt.xlabel("Patient ID (Sorted by Median Saliency)")
plt.ylabel("Normalized Saliency")
plt.title("Saliency Distribution Across Patients (Ordered by Median)")
plt.grid(axis="y", alpha=0.7)


# Save and display the plot
plt.savefig('/tudelft.net/staff-umbrella/EleniSalient/Results/Saliency_maps/saliency_distribution_pp_boxplot.png')

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # # Find the peak segment index (adjusted for skipping segments)
    # peak_segment_index = torch.argmax(normalized_saliency).item() 
    # print(peak_segment_index)
    
    # Plot the saliency map for the first patient as an example
    # if i >= 10 and i < 20:
    #     plot_saliency_map(normalized_saliency.numpy(), f'patient_{grouped_patients[i][0]}_10.png', title='Saliency Map')

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # So we need to compute saliency and stuff with the predictions and not during the epochs

# # Analyze entropy changes for saliency
# for epoch_entropy in entropy_history:
#     delta_entropy = epoch_entropy[:, 1:] - epoch_entropy[:, :-1]  # Compute changes in entropy
#     significant_changes = (delta_entropy.abs() > 0.1).nonzero(as_tuple=True)
#     print("Significant entropy changes found at segments:", significant_changes[1])

