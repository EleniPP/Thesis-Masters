import torch
import matplotlib.pyplot as plt
import numpy as np


def entropy(probabilities):
    return -(probabilities * torch.log(probabilities)).sum(dim=-1)

def plot_saliency_map(saliency_values, title='Saliency Map'):
    plt.figure(figsize=(10, 5))
    plt.plot(saliency_values, label='Saliency')
    plt.xlabel('Segment')
    plt.ylabel('Normalized Saliency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize_saliency(saliency_values):
    max_val = torch.max(saliency_values)
    min_val = torch.min(saliency_values)
    normalized_saliency = (saliency_values - min_val) / (max_val - min_val)
    return normalized_saliency

# Load the tensor from the file
probability_distributions = torch.load('probability_distributions.pth')
all_patient_numbers = torch.load('all_patient_numbers.pth')
# Now `probability_distributions` is ready for use
print(probability_distributions.shape)  # Should print torch.Size([28317, 2])
torch.set_printoptions(threshold=torch.inf)
print(all_patient_numbers)


# Compute the entropy for each probability distribution (each row in the tensor)
segment_entropies = entropy(probability_distributions)

# Convert entropies to NumPy for gradient calculation
segment_entropies_np = segment_entropies.numpy()

# Compute the gradient of entropy across segments (not per single segment)
jacobian = np.gradient(segment_entropies_np)

# Since this is a 1D problem, J'(x)J(x) is 1x1, therefore determinant
# is the square of gradient itself
segment_saliency = np.square(jacobian)

# Normalize saliency
normalized_saliency = normalize_saliency(torch.tensor(segment_saliency))

# Plot the saliency map
plot_saliency_map(normalized_saliency.numpy(), title='Saliency Map')
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

