import torch
import numpy as np
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import csv
from temperature_scaling import ModelWithTemperature
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.utils import resample
import torch.optim.lr_scheduler as lr_scheduler


def flatten(multimodal_features, labels):
    flattened_features = []
    flattened_labels = []
    num_seg_per_patient = []
    segment_indices = []  # To track original chronological order within each patient
    for i, (features, label) in enumerate(zip(multimodal_features, labels)):
        cnt = 0
        for idx, segment in enumerate(features):
            cnt += 1
            flattened_features.append(segment)
            flattened_labels.append(label)
            segment_indices.append(idx)
        num_seg_per_patient.append(cnt)
    # Convert flattened lists to tensors
    flattened_features = torch.stack(flattened_features)
    flattened_labels = torch.tensor(flattened_labels)
    segment_per_patient = torch.tensor(num_seg_per_patient)
    segment_indices = torch.tensor(segment_indices)  # Contains the index of each segment for all the segments in the train multimodal
    return flattened_features, flattened_labels, segment_per_patient, segment_indices

def get_split(split):
    base_path = "/tudelft.net/staff-umbrella/EleniSalient/"
    extension = "_split.csv"
    file = f"{base_path}{split}{extension}"
    patients_list = []
    max_row = 47
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(csvfile)
        row_count = 0
        for row in spamreader:
            if (row_count >= max_row) and (split == "test"):
                break
            if row[0] == '402' or row[0] == '420':
                row_count += 1
                continue
            patients_list.append(row[0])
            row_count += 1
    patients_array = np.array(patients_list).astype(np.int32)
    return patients_array

def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    threshold = 0.5
    with torch.no_grad():  # Disable gradient calculation
        for features, labels, _, _ in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs,_ ,_ = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
            predicted = (probs > threshold).int()
            all_predictions.append(predicted)
            all_labels.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = correct / total if total > 0 else 0
    print(classification_report(all_labels, all_predictions))
    return test_loss / len(dataloader), accuracy, all_predictions, all_labels

def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for features, labels,_,_ in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs, _ , _= model(features)
            loss = criterion(outputs.view(-1, 2), labels.view(-1))
            val_loss += loss.item()
    return val_loss / len(dataloader)

def filter_by_patient_numbers(features, labels, patient_numbers):
    indices = [patient_to_index[patient] for patient in patient_numbers if patient in patient_to_index] # Get indices of the selected patients
    print(f"Selected {len(indices)} patients out of {len(patient_numbers)}")
    filtered_features = [features[idx] for idx in indices]
    filtered_labels = [labels[idx] for idx in indices]
    return filtered_features, filtered_labels


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads."

        # Linear layers for each head
        self.query_weights = nn.ModuleList([nn.Linear(self.head_dim, 1) for _ in range(num_heads)])
        self.output_layer = nn.Linear(input_dim, input_dim)  # Combine all heads into the same input_dim

    def forward(self, x):
        """
        x: [batch_size, input_dim] - Independent segments
        """
        batch_size, input_dim = x.shape

        # Split input into heads: [batch_size, num_heads, head_dim]
        x_split = x.view(batch_size, self.num_heads, self.head_dim)

        # Compute attention scores for each head
        head_scores = []
        entropy_loss = 0
        for i in range(self.num_heads):
            scores = torch.sigmoid(self.query_weights[i](x_split[:, i, :]))  # [batch_size, 1]
            head_scores.append(scores)

            # Compute entropy for each head
            entropy_loss += -torch.sum(scores * torch.log(scores + 1e-8)) / batch_size

        # Concatenate scores: [batch_size, num_heads]
        head_scores = torch.cat(head_scores, dim=-1)

        # Apply attention scores to each head
        attended_heads = x_split * head_scores.unsqueeze(-1)  # [batch_size, num_heads, head_dim]

        # Combine all heads: [batch_size, input_dim]
        attended_output = attended_heads.view(batch_size, -1)

        # Final transformation to aggregate head outputs
        attended_output = self.output_layer(attended_output)

        return attended_output, head_scores, entropy_loss

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)  # Linear layer to compute attention scores

    def forward(self, x):
        """
        x: [batch_size, num_segments, input_dim] - Input features
        """
        scores = self.attention_weights(x)  # Compute attention scores: [batch_size, 1]
        scores = torch.sigmoid(scores)  # Normalize scores to [0, 1]
        attended_output = x * scores  # Apply attention scores to each segment
        return attended_output, scores  # Return the weighted features and attention scores


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            ce_loss = at * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DepressionDatasetCross(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DepressionDataset(Dataset):
    def __init__(self, data, labels, patient_numbers, segment_indices):
        self.data = data
        self.labels = labels
        self.patient_numbers = patient_numbers
        self.segment_indices = segment_indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.patient_numbers[idx], self.segment_indices[idx]

def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    return data_padded, labels_padded

class DepressionPredictor1(nn.Module):
    def __init__(self):
        super(DepressionPredictor1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        # self.attention = AttentionLayer(512) # Attention layer
        self.attention = MultiHeadAttentionLayer(256, num_heads=4)  # Multi-head attention layer
        self.classifier2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
         x: [batch_size, input_dim] - Independent segments in the batch.
        """
        # Encode each segment
        encoded_segments = self.classifier(x)  # [batch_size, 512]

        # Apply attention
        attended_segments, attention_scores, attention_entropy= self.attention(encoded_segments)  # [batch_size, 512], [batch_size, 1]

        # Classify each segment
        logits = self.classifier2(attended_segments)  # [batch_size, 2]
        return logits, attention_scores, attention_entropy
    # def forward(self, x):
    #     x = self.classifier(x)
    #     if torch.isnan(x).any() or torch.isinf(x).any():
    #         print("Found nan or inf in model output")
    #     return x

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/loss_plot_imbalanced_focal.png')
    plt.close()

def visualize_attention_scores2(attention_scores_per_epoch):
    """
    Visualizes attention scores from the new attention mechanism.

    Args:
        attention_scores_per_epoch: List of attention scores for each epoch.
    """
    # Extract attention scores from the first epoch
    first_epoch_attention = attention_scores_per_epoch[0]  # First epoch
    all_attention_scores = np.concatenate([batch.squeeze() for batch in first_epoch_attention], axis=0)

    # Plot histogram of attention scores
    plt.hist(all_attention_scores, bins=50, color='blue', alpha=0.7)
    plt.xlabel("Attention Score")
    plt.ylabel("Frequency")
    plt.title("Attention Score Distribution (First Epoch)")
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/attention_score_distribution.png')
    plt.close()

    # Visualize attention scores for the first batch
    first_batch_attention = first_epoch_attention[0].squeeze()  # Shape: [batch_size]
    plt.plot(first_batch_attention, marker='o', linestyle='-', color='red')
    plt.xlabel("Segment Index")
    plt.ylabel("Attention Score")
    plt.title("Attention Scores for First Batch (First Epoch)")
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/attention_scores_first_batch.png')
    plt.close()

    print("Attention score visualizations saved.")

def visualize_attention_scores(attention_scores):
    # Select the first head's scores
    first_head_scores = attention_scores[:, 0]  # Shape: [batch_size]

    # Plot scores
    plt.hist(first_head_scores.detach().cpu().numpy(), bins=50, alpha=0.7, label='First Head')
    plt.xlabel("Attention Score")
    plt.ylabel("Frequency")
    plt.title("Attention Scores for First Head")
    plt.legend()
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/_multihead_attention_scores_first_batch.png')
    plt.close()


def train_model(model, dataloader, val_loader, optimizer, scheduler, criterion, epochs=30):
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    lambda_entropy = 0.01  # Entropy regularization weight  
    attention_scores_per_epoch = []  # To store attention scores for each epoch
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_attention_scores = []  # Store attention scores for this epoch
        for i, (features, labels,_,_) in enumerate(dataloader):
            optimizer.zero_grad()
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs, attention_scores, attention_entropy= model(features)
            loss = criterion(outputs, labels)

            # Add entropy regularization
            total_loss_with_entropy = loss + lambda_entropy * attention_entropy
            # # Apply attention-based weighting
            # attention_weights = attention_scores.squeeze()  # Shape: [batch_size]
            # normalized_attention_weights = attention_weights / attention_weights.sum()  # Normalize scores
            # clipped_attention_weights = torch.clamp(normalized_attention_weights, min=0.1)
            # loss = (clipped_attention_weights * loss).mean()  # Weight loss by attention scores
            total_loss_with_entropy.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += total_loss_with_entropy.item()

            # Store attention scores for this batch
            epoch_attention_scores.append(attention_scores.detach().cpu().numpy())

        attention_scores_per_epoch.append(epoch_attention_scores)  # Save for the epoch

        val_loss = validate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
        train_losses.append(total_loss / len(dataloader))
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss / len(dataloader)}, Validation Loss: {val_loss}')
    plot_losses(train_losses, val_losses)
    return model, attention_scores_per_epoch

if __name__ == "__main__":
    # Load the audio tensor
    # audio_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/audio_features.npy', allow_pickle=True)
    audio_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/audio_features_reduced_reliable_prep.npy', allow_pickle=True)
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Audio feature sample shape: {audio_features[0].shape}")

    # Load the visual tensor
    # visual_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/extracted_visual_features.npy', allow_pickle=True)
    visual_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/extracted_visual_features_reduced_reliable_prep.npy', allow_pickle=True)
    print(f"Visual features shape: {visual_features.shape}")
    print(f"Visual feature sample shape: {visual_features[0].shape}")

    # Load the labels
    with open('/tudelft.net/staff-umbrella/EleniSalient/labels.json', 'r') as file:
        labels_dict = json.load(file)
    del labels_dict['492']

    numbers = [int(key) for key in labels_dict.keys()]
    labels = list(labels_dict.values())
    tlabels = torch.tensor(labels)

    # Convert audio and visual features to torch tensors
    audio_features = [torch.from_numpy(a) for a in audio_features]
    visual_features = [torch.from_numpy(v) for v in visual_features]

    for i, (a, v) in enumerate(zip(audio_features, visual_features)):
        if a.shape[0] != v.shape[0]:
            print(f"Mismatch in number of segments for index {i}: audio {a.shape[0]}, visual {v.shape[0]}")

    # Concatenate audio and visual features for each entry
    multimodal_features = [torch.cat((a, v), dim=1) for a, v in zip(audio_features, visual_features)]

    # Normalize each tensor individually
    normalized_multimodal = []
    for tensor in multimodal_features:
        mean = tensor.mean(dim=0)
        std = tensor.std(dim=0)
        normalized_tensor = (tensor - mean) / (std + 1e-5)
        normalized_multimodal.append(normalized_tensor)

    # Identify and remove NaN entries and flatten inputs for training
    cleaned_multimodal_features = []
    cleaned_labels = []
    for i, (features, label) in enumerate(zip(normalized_multimodal, labels)):
        if not torch.isnan(features).any() and not torch.isinf(features).any():
            cleaned_multimodal_features.append(features)
            cleaned_labels.append(label)
        else:
            print(f"Removed entry with NaN/Inf values at index {i}")

    # Re-assign the cleaned data
    normalized_multimodal = cleaned_multimodal_features
    tlabels = torch.tensor(cleaned_labels)

    # Get splits
    train_split = get_split('train')
    test_split = get_split('test')
    val_split = get_split('dev')
    train_split = train_split[:-1]

    del numbers[117]
    del numbers[99]
    patient_to_index = {patient_num: idx for idx, patient_num in enumerate(numbers)}

    for i, features in enumerate(normalized_multimodal):
        if torch.isnan(features).any():
            print(f"NaN values in normalized_multimodal before dataloaders at index {i}")



    train_multimodal, train_labels = filter_by_patient_numbers(normalized_multimodal, labels, train_split)
    train_multimodal, train_labels, segments_per_patient_train, segments_order_train = flatten(train_multimodal, train_labels)
    print(train_multimodal.shape)
    print(train_multimodal[0].shape)
    # #Check inbalance
    unique, counts = np.unique(train_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    print(f"Class distribution: {class_distribution}")

    # Determine majority and minority class
    majority_class = max(class_distribution, key=class_distribution.get)
    minority_class = min(class_distribution, key=class_distribution.get)


    # Get indices of the majority and minority classes
    majority_indices = np.where(train_labels == majority_class)[0]
    minority_indices = np.where(train_labels == minority_class)[0]

    n_majority = len(majority_indices)
    n_minority = len(minority_indices)


    # #MAKE BALANCED MINI BATCHES
    # class_counts = np.bincount(train_labels)
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # sample_weights = class_weights[train_labels]  # Assuming `labels` contains class labels
    # # sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) 
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # TRY THE 50-50 DISTRIBUTION

    # Undersample the majority class to match the number of minority cl   ass samples
    majority_undersampled_indices = resample(majority_indices, 
                                             replace=False,  # No replacement, undersample without duplication
                                             n_samples=n_minority,  # Match the number of minority class samples
                                             random_state=42)

    # Combine the minority and undersampled majority indices
    balanced_indices = np.concatenate([majority_undersampled_indices, minority_indices])

    # Shuffle the indices to ensure randomization
    np.random.shuffle(balanced_indices)

    balanced_indices = torch.tensor(balanced_indices, dtype=torch.long)  # Convert indices to tensor if not already

    # Index train_multimodal and train_labels using balanced_indices
    train_multimodal = train_multimodal[balanced_indices]
    train_labels = train_labels[balanced_indices]



    # Check the new class distribution
    unique_balanced, counts_balanced = np.unique(train_labels, return_counts=True)
    class_distribution_balanced = dict(zip(unique_balanced, counts_balanced))
    print(f"Balanced class distribution: {class_distribution_balanced}")
    # array that has the patient number for each segment in the train multimnodal
    segments_patients_train = [num for count, num in zip(segments_per_patient_train, train_split) for _ in range(count)]

    # IMBALANCED DATASET
    # segments_patients_train = [num for count, num in zip(segments_per_patient_train, train_split) for _ in range(count)]

    


    val_multimodal, val_labels = filter_by_patient_numbers(normalized_multimodal, labels, val_split)
    val_multimodal, val_labels, segments_per_patient_val, segments_order_val = flatten(val_multimodal, val_labels)
    segments_patients_val = [num for count, num in zip(segments_per_patient_val, val_split) for _ in range(count)]

    test_multimodal, test_labels = filter_by_patient_numbers(normalized_multimodal, labels, test_split)
    test_multimodal, test_labels, segments_per_patient_test, segments_order_test = flatten(test_multimodal, test_labels)
    segments_patients_test = [num for count, num in zip(segments_per_patient_test, test_split) for _ in range(count)]

    train_dataset = DepressionDataset(train_multimodal, train_labels, segments_patients_train, segments_order_train)
    val_dataset = DepressionDataset(val_multimodal, val_labels, segments_patients_val, segments_order_val)
    test_dataset = DepressionDataset(test_multimodal, test_labels, segments_patients_test, segments_order_test)

    train_loader = DataLoader(train_dataset, batch_size=128,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = DepressionPredictor1()
    # why no SGD?
    optimizer = optim.Adam(model.parameters(),  lr=1e-4, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    alpha = torch.tensor([1.0, 2.0])
    criterion = FocalLoss(alpha=alpha, gamma=1.6)
    # criterion = nn.CrossEntropyLoss()

    model, attention_scores_per_epoch = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion)

    # Evaluate on the validation set
    val_loss, accuracy, _, _ = evaluate_model(model, val_loader, criterion)
    # val_loss, accuracy, _, _ = evaluate_model(model, train_loader, criterion)
    visualize_attention_scores(attention_scores_per_epoch)