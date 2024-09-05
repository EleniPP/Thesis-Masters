import torch
import matplotlib.pyplot as plt
import numpy as np


def entropy(probabilities):
    return -(probabilities * torch.log(probabilities)).sum(dim=-1)

def plot_saliency_map(saliency_values, file_name,  title='Saliency Map', save=True):
    plt.figure(figsize=(10, 5))
    plt.plot(saliency_values, label='Saliency')
    plt.xlabel('Segment')
    plt.ylabel('Normalized Saliency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig(f'D:/Results/Saliency_maps/{file_name}')
        plt.close()
    else:
        plt.show()


def normalize_saliency(saliency_values):
    max_val = torch.max(saliency_values)
    min_val = torch.min(saliency_values)
    normalized_saliency = (saliency_values - min_val) / (max_val - min_val)
    return normalized_saliency

# Load the tensor from the file
probability_distributions = torch.load('probability_distributions.pth')
print(probability_distributions[0].type)
all_patient_numbers = torch.load('all_patient_numbers.pth')
all_segment_orders = torch.load('all_segment_orders.pth')


# now this one has shuffled patient numbers. Each patient number appears in the torch as many times as the segments the patient video has. Each patient number
# is in the position of the list where the respective segment is on the probability distribution list of the segment.
torch.set_printoptions(threshold=torch.inf)

# So what i want is for each patient, to have a probability distribution so i can calculate the jacobian and the saliency 
# So first we have to align again the probability distribution torch with the all patient numbers torch after we put in ascending order the all patient numbers torch.
sorted_patients, indices = torch.sort(all_patient_numbers)
# Then we use the indices to sort the probability distributions in the same order as the patients were sorted so that each segment belongs to the right patient
sorted_probability_distributions = probability_distributions[indices]
sorted_segment_orders = all_segment_orders[indices]

# Now create a tensor of probability distributions for each patient. So i need to group sorted patients by patient.
grouped_patients = []
grouped_probabilities = []

# get unique values and how many times they appear
unique_values, counts = torch.unique(sorted_patients, return_counts=True)

# to keep track of where the groups start
start_idx = 0
# Iterate through each unique value
for value, count in zip(unique_values, counts):
    patients=sorted_patients[start_idx:start_idx + count] #each patients tensor contains the id of a patient #ofsegments times
    patient_segment_orders = sorted_segment_orders[start_idx:start_idx + count] #each patient_segment_orders contain the index of each segment which is the index from its original order before the shuffling
    probabilities = sorted_probability_distributions[start_idx:start_idx + count] #each probabilities tensor contains the probability distribution of the segment in the aligned position with the above.

    # Sort the segment orders to get the chronological order
    sorted_orders, sorted_order_indices = torch.sort(patient_segment_orders)  # Sort the orders and get indices
    # And now we also parallely sort the probabilities so they will also be in chronological order and that each probability will be alignes in its segment
    sorted_probabilities = probabilities[sorted_order_indices]

    # Slice the tensors based on the count of each unique value
    grouped_patients.append(sorted_patients[start_idx:start_idx + count]) #slice starts at start_idx and ends at start_idx+count
    grouped_probabilities.append(sorted_probabilities)

    # Update the starting index for the next group
    start_idx += count


print(type(grouped_probabilities))
print(type(grouped_probabilities[0]))
# so now grouped probabilities is a list of torches

for i,probability_distribution in enumerate(grouped_probabilities):

    # Compute the entropy for each probability distribution (each row in the tensor)
    entropies = entropy(probability_distribution)

    # Convert entropies to NumPy for gradient calculation
    entropies_np = entropies.numpy()

    # Compute the gradient of entropy across segments (not per single segment)
    jacobian = np.gradient(entropies_np)

    # Since this is a 1D problem, J'(x)J(x) is 1x1, therefore determinant
    # is the square of gradient itself
    patient_saliency = np.square(jacobian)

    # Normalize saliency
    normalized_saliency = normalize_saliency(torch.tensor(patient_saliency))

    peak_segment_index = torch.argmax(normalized_saliency).item()
    print(peak_segment_index)
    # Plot the saliency map
    # plot_saliency_map(normalized_saliency.numpy(),f'patient_{grouped_patients[i][0]}.png', title='Saliency Map')
# --------------------------------------------------------------------------------------

# # comment for the first run / maybe i'll put it in a different file
# for probabilities in probability_distributions:
#     segment_entropy = entropy(probabilities).numpy().flatten()
#     # print(entropies.shape)
#     # Compute gradients
#     jacobian = np.gradient(segment_entropy)
#     # Since this is a 1d problem, J'(x)J(x) is 1x1, therefore determinant
#     # is the square of gradient itself.
#     segment_saliency = np.square(jacobian)
#     # print(jacobian)
#     plot_saliency_map(segment_saliency, title=f'Saliency Map')
#     # saliency = saliency_from_jacobian(jacobian)
#     # print(saliency)



#         normalized_saliency = normalize_saliency(saliency)

#         # Debug prints to verify dimensions and values
#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')
#         print(f'Logits shape: {outputs.shape}')
#         print(f'Probabilities shape: {probabilities.shape}')
#         print(f'Entropy shape: {entropy(probabilities).shape}')
#         print(f'Jacobian matrix shape: {jacobian_matrix.shape}')
#         print(f'Saliency shape: {saliency.shape}')
#         print(f'Normalized Saliency shape: {normalized_saliency.shape}')
#         print('NORMALIZED SALIENCY')
#         print( saliency)

#         # Plot saliency map for each epoch (optional)
#         plot_saliency_map(normalized_saliency, title=f'Saliency Map - Epoch {epoch+1}')
# # ------------------------------------------------------------------------------------------------
# # So we need to compute saliency and stuff with the predictions and not during the epochs

# # Analyze entropy changes for saliency
# for epoch_entropy in entropy_history:
#     delta_entropy = epoch_entropy[:, 1:] - epoch_entropy[:, :-1]  # Compute changes in entropy
#     significant_changes = (delta_entropy.abs() > 0.1).nonzero(as_tuple=True)
#     print("Significant entropy changes found at segments:", significant_changes[1])

