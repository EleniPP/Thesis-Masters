import torch
import matplotlib.pyplot as plt
import numpy as np


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

print('Loaded tensors shapes:')
print(all_patient_numbers.shape)
print(all_segment_orders.shape)
print(predictions.shape)
print(true_labels.shape)
print(probability_distributions.shape)

# # Load the tensor from the file // Previous version
# probability_distributions2 = torch.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/probability_distributions.pth')
# print(probability_distributions[0].type)
# all_patient_numbers2 = torch.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/all_patient_numbers.pth')
# all_segment_orders2 = torch.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/all_segment_orders.pth')


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
global_threshold = 0.3178  # Set to the global 90th percentile
# threshold = 0.8  # Set to the local threshold
all_normalized_saliencies = []
# Open two files for writing salient indices and correctness
# with open("salient_indices.txt", "w") as indices_file, open("salient_correctness.txt", "w") as correctness_file:
all_entropies = []
for i, probability_distribution in enumerate(grouped_probabilities):

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

    # Apply global threshold
    filtered_saliency = normalized_saliency[normalized_saliency > global_threshold]

    if filtered_saliency.size(0) > 0:
        # Apply dynamic threshold (e.g., 99th percentile) within the filtered saliencies
        dynamic_threshold = torch.quantile(filtered_saliency, 0.8)
        final_saliency = filtered_saliency[filtered_saliency > dynamic_threshold]
        salient_indices = torch.nonzero(filtered_saliency > dynamic_threshold).view(-1).tolist()

        # Get classification correctness for salient indices
        salient_correctness = []
        for idx in salient_indices:
            true_label = grouped_true_labels[i][idx].item()  # True label of the salient segment
            prediction = grouped_predictions[i][idx].item()  # Model's prediction for the salient segment
            correctness = (true_label == prediction)  # True if correctly classified
            salient_correctness.append(correctness)

        # Get the total number of segments for the patient
        total_segments = len(grouped_probabilities[i])  # Number of segments for this patient

    # Save results to files
        # indices_file.write(f"Patient {i}: {salient_indices}, Total Segments: {total_segments}\n")
        # indices_file.write(f"Patient {i}: {salient_indices}\n")
        # correctness_file.write(f"Patient {i}: {salient_correctness}\n")
        print(f"Patient {i}, Final Selected Saliencies: {salient_indices}")
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


# Convert to numpy array for easier analysis
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

