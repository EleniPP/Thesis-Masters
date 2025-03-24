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
from torch.utils.data import random_split, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from collections import Counter

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
            if row[0] == '422':
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
        for features, labels,_,_ in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
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
            outputs = model(features)
            loss = criterion(outputs.view(-1, 2), labels.view(-1))
            val_loss += loss.item()
    return val_loss / len(dataloader)

def filter_by_patient_numbers(features, labels, patient_numbers):
    indices = [patient_to_index[patient] for patient in patient_numbers if patient in patient_to_index]
    filtered_features = [features[idx] for idx in indices]
    filtered_labels = [labels[idx] for idx in indices]
    return filtered_features, filtered_labels

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
        print('Inside DepressionDataset')
        print(f"Data shape: {len(self.data)}, Labels shape: {len(self.labels)}, Patient numbers: {len(self.patient_numbers)}, Segment indices: {len(self.segment_indices)}")

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
            # nn.Linear(17408, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.classifier(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found nan or inf in model output")
        return x

# def plot_losses(train_losses, val_losses):
def plot_losses(train_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/loss_plot_only_train.png')
    plt.close()

def train_model(model, dataloader, optimizer, scheduler, criterion, epochs=100):
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (features, labels,_,_) in enumerate(dataloader):
            optimizer.zero_grad()
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        # val_loss = validate_model(model, val_loader, criterion)
        # scheduler.step(val_loss)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        # if patience_counter >= early_stopping_patience:
        #     print("Early stopping triggered")
        #     break
        train_losses.append(total_loss / len(dataloader))
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(dataloader)}")
        # val_losses.append(val_loss)
        # print(f'Epoch {epoch+1}, Training Loss: {total_loss / len(dataloader)}, Validation Loss: {val_loss}')
    plot_losses(train_losses)
    torch.save(model.state_dict(), '../../../tudelft.net/staff-umbrella/EleniSalient/final_model1.pth')
    return model

def expected_calibration_error(y_true, y_probs, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_mask = (y_probs >= bin_edges[i]) & (y_probs < bin_edges[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            avg_prob = np.mean(y_probs[bin_mask])
            avg_true = np.mean(y_true[bin_mask])
            ece += (bin_count / len(y_true)) * np.abs(avg_prob - avg_true)
    return ece


if __name__ == "__main__":
    # Load the audio tensor
    # audio_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/audio_features.npy', allow_pickle=True)
    audio_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/audio_features_reduced_reliable.npy', allow_pickle=True)
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Audio feature sample shape: {audio_features[0].shape}")

    # Load the visual tensor
    # visual_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/extracted_visual_features.npy', allow_pickle=True)
    visual_features = np.load('/tudelft.net/staff-umbrella/EleniSalient/Preprocessing/extracted_visual_features_reduced_reliable.npy', allow_pickle=True)
    print(f"Visual features shape: {visual_features.shape}")
    print(f"Visual feature sample shape: {visual_features[0].shape}")

    # Load the labels
    with open('/tudelft.net/staff-umbrella/EleniSalient/labels.json', 'r') as file:
        labels_dict = json.load(file)
    del labels_dict['492']

    # Contains 402 and 420
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
    # print(f"Test split: {test_split}")
    # print(test_split.shape)
    val_split = get_split('dev')
    # print(f"Val split: {val_split}")
    # print(val_split.shape)
    train_split = train_split[:-1]
    # print(f"Train split: {train_split}")
    # print(train_split.shape)

    # del numbers[117]
    # del numbers[99]
    patient_to_index = {patient_num: idx for idx, patient_num in enumerate(numbers)}

    for i, features in enumerate(normalized_multimodal):
        if torch.isnan(features).any():
            print(f"NaN values in normalized_multimodal before dataloaders at index {i}")

    # train_multimodal, train_labels = filter_by_patient_numbers(normalized_multimodal, labels, train_split)
    # train_multimodal, train_labels, segments_per_patient_train, segments_order_train = flatten(train_multimodal, train_labels)
    # print(train_multimodal.shape)
    # print(train_multimodal[0].shape)
    # # #Check inbalance
    # unique, counts = np.unique(train_labels, return_counts=True)
    # class_distribution = dict(zip(unique, counts))

    # print(f"Class distribution: {class_distribution}")

    # # Determine majority and minority class
    # majority_class = max(class_distribution, key=class_distribution.get)
    # minority_class = min(class_distribution, key=class_distribution.get)


    # # Get indices of the majority and minority classes
    # majority_indices = np.where(train_labels == majority_class)[0]
    # minority_indices = np.where(train_labels == minority_class)[0]

    # n_majority = len(majority_indices)
    # n_minority = len(minority_indices)


    # #MAKE BALANCED MINI BATCHES
    # class_counts = np.bincount(train_labels)
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    # sample_weights = class_weights[train_labels]  # Assuming `labels` contains class labels
    # # sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) 
    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # TRY THE 50-50 DISTRIBUTION

    # # Undersample the majority class to match the number of minority cl   ass samples
    # majority_undersampled_indices = resample(majority_indices, 
    #                                          replace=False,  # No replacement, undersample without duplication
    #                                          n_samples=n_minority,  # Match the number of minority class samples
    #                                          random_state=42)

    # # Combine the minority and undersampled majority indices
    # balanced_indices = np.concatenate([majority_undersampled_indices, minority_indices])

    # # Shuffle the indices to ensure randomization
    # np.random.shuffle(balanced_indices)

    # balanced_indices = torch.tensor(balanced_indices, dtype=torch.long)  # Convert indices to tensor if not already

    # # Index train_multimodal and train_labels using balanced_indices
    # train_multimodal = train_multimodal[balanced_indices]
    # train_labels = train_labels[balanced_indices]


    # # Check the new class distribution
    # unique_balanced, counts_balanced = np.unique(train_labels, return_counts=True)
    # class_distribution_balanced = dict(zip(unique_balanced, counts_balanced))
    # print(f"Balanced class distribution: {class_distribution_balanced}")
    # # array that has the patient number for each segment in the train multimnodal
    # segments_patients_train = [num for count, num in zip(segments_per_patient_train, train_split) for _ in range(count)]
# ---------------------------------------------------------
    # Mexri tora ola mou ta features exoun to sosto number of segments!!!! Yeyyyyyyy
    train_multimodal, train_labels, segments_per_patient_train, segments_order_train = flatten(normalized_multimodal, cleaned_labels)
    # TESTED it is the correct number of segments per patient
    # print(segments_per_patient_train)
    # print('END')

    combined_split = np.concatenate((train_split, test_split, val_split))
    # Optional: Sort the combined array if needed
    patients = np.sort(combined_split)
    print(patients)
    segments_patients_train = [patient for count, patient in zip(segments_per_patient_train, patients) for _ in range(count)]
    # Count occurrences of each patient ID in segments_patients_train
    segment_counts = Counter(segments_patients_train)

    # Print the number of segments per patient
    for patient_id, count in segment_counts.items():
        print(f"Patient {patient_id}: {count}")

# --------------------------------------------------------- 
# # MORE TRAINING SET
#     # Split at the patient level for callibration set (because before I was splitting after so there were segments from the same patient that
#     # were in calibration set and others that was in the training set /  this was bad because when i checked the entropy changes i did it for all the segments of the patient
#     # but some of them were in the calibration set))
#     num_patients = len(normalized_multimodal) # Total number of patients
#     print(f"Total number of patients: {num_patients}")
#     train_patients, calibration_patients, train_labels, calibration_labels = train_test_split(
#         np.arange(num_patients), cleaned_labels, test_size=0.1, random_state=42, stratify=cleaned_labels
#     )

#     # Convert train and calibration patient indices to Python lists
#     train_patients = train_patients.tolist()
#     calibration_patients = calibration_patients.tolist()

#     # Index normalized_multimodal using list comprehension
#     train_multimodal = [normalized_multimodal[i] for i in train_patients]
#     calibration_multimodal = [normalized_multimodal[i] for i in calibration_patients]

# ---------------------------------------------------------
    # print('Shapes after the split')
    # # IMBALANCED DATASET
    # train_multimodal, train_labels, segments_per_patient_train, segments_order_train = flatten(train_multimodal, train_labels)
    # print(train_labels.shape)
    # print(train_multimodal.shape)


    # # segments_patients_train = [num for count, num in zip(segments_per_patient_train, train_split) for _ in range(count)]
    # segments_patients_train = [patient for count, patient in zip(segments_per_patient_train, train_patients) for _ in range(count)]


    # calibration_multimodal, calibration_labels, segments_per_patient_calibration, segments_order_calibration = flatten(calibration_multimodal, calibration_labels)
    # print(calibration_labels.shape)
    # print(calibration_multimodal.shape)
    # segments_patients_calibration = [patient for count, patient in zip(segments_per_patient_calibration, calibration_patients) for _ in range(count)]


    train_dataset = DepressionDataset(train_multimodal, train_labels, segments_patients_train, segments_order_train)
    # calibration_dataset = DepressionDataset(calibration_multimodal, calibration_labels, segments_patients_calibration, segments_order_calibration)

# -------------------------------------------------------
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    calibration_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    # calibration_loader = DataLoader(calibration_dataset, batch_size=64, shuffle=False)
# -------------------------------------------------------

    model = DepressionPredictor1()
    # why no SGD?
    optimizer = optim.Adam(model.parameters(),  lr=1e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    alpha = torch.tensor([1.0, 2.0])
    criterion = FocalLoss(alpha=alpha, gamma=1.6)
    # criterion = nn.CrossEntropyLoss()

    model = train_model(model, train_loader, optimizer, scheduler, criterion)

    # Evaluate on the validation set
    # val_loss, accuracy, _, _ = evaluate_model(model, val_loader, criterion)
    val_loss, accuracy, _, _ = evaluate_model(model, train_loader, criterion)

    # Load the saved model
    model = DepressionPredictor1()  # Initialize your model architecture
    model.load_state_dict(torch.load('../../../tudelft.net/staff-umbrella/EleniSalient/final_model1.pth'))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded and ready for calibration")


# --------------------------------------------------- Try another calibration method ------------------------------------
    def platt_scaling(logits, labels):
        """
        Apply Platt Scaling to calibrate the model probabilities.

        Args:
            logits (np.array): Logits from the model, shape [num_samples, num_classes].
            labels (np.array): True labels, shape [num_samples].

        Returns:
            LogisticRegression: Trained logistic regression model.
            np.array: Calibrated probabilities.
        """
        # Ensure logits are numpy arrays
        logits = logits.numpy() if not isinstance(logits, np.ndarray) else logits
        labels = labels.numpy() if not isinstance(labels, np.ndarray) else labels
        
        # Use only the logit for class 1 (binary classification)
        class_1_logits = logits[:, 1]
        
        # Fit logistic regression
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(class_1_logits.reshape(-1, 1), labels)
        
        # Predict calibrated probabilities
        calibrated_probs = lr.predict_proba(class_1_logits.reshape(-1, 1))[:, 1]
        
        return lr, calibrated_probs


        # Initialize results dictionary
    calibration_results = {
        "patient_ids": [],
        "segment_indices": [],
        "predictions": [],
        "true_labels": [],
        "logits": [],
        "calibrated_probs": [],
        "calibrated_probs_platt": []
    }

    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(calibration_loader)

    scaled_model.eval()

    all_probs = []
    all_patient_numbers = []
    all_segment_orders = []
    with torch.no_grad():   
        for inputs, labels , patient_numbers,segments_order in calibration_loader:
            logits = model(inputs)  # Model logits
            probs = torch.softmax(logits, dim=1)  # Calibrated probabilities
            predictions = (probs[:, 1] > 0.5).int()  # Binary predictions for class 1

            # Save results
            calibration_results["patient_ids"].extend(patient_numbers.tolist())
            calibration_results["segment_indices"].extend(segments_order.tolist())
            calibration_results["predictions"].extend(predictions.tolist())
            calibration_results["true_labels"].extend(labels.tolist())
            calibration_results["logits"].extend(logits.tolist())  # Store entire probability distribution
            calibration_results["calibrated_probs"].extend(probs.tolist())  # Store entire probability distribution

    probs = torch.tensor(calibration_results["calibrated_probs"])[:, 1].numpy()  # Probabilities for class 1
    # probs = torch.tensor(calibration_results["calibrated_probs"]).numpy()  # Probabilities for class 1
    true_labels = torch.tensor(calibration_results["true_labels"]).numpy()  # True labels
    # Reliability Diagram before Calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, probs, n_bins=10)

    # Plot Reliability Diagram
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Platt Calibrated Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Value")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/calibration_curve_before.png')
    plt.close()

    # CALIBRATION
    # Extract probabilities for class 1 and true labels
    logits = torch.tensor(calibration_results["logits"]).numpy()  # Probabilities for class 1
    true_labels = torch.tensor(calibration_results["true_labels"]).numpy()  # True labels

    # Apply Platt Scaling
    platt_model, calibrated_probs = platt_scaling(logits, true_labels)

    # Convert calibrated_probs to two-class probabilities
    calibrated_probs_two_class = np.column_stack((1 - calibrated_probs, calibrated_probs))
    calibration_results["calibrated_probs_platt"] = calibrated_probs_two_class.tolist()
    # Save calibrated probabilities into the calibration_results dictionary
    # calibration_results["calibrated_probs"] = calibrated_probs.tolist()  # Store calibrated probabilities

    # Evaluate the Calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, calibrated_probs, n_bins=10)

    # Plot Reliability Diagram after calibration
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Platt Calibrated Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Value")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/calibration_curve.png')
    plt.close()

    # Compute ECE
    ece_calibrated = expected_calibration_error(true_labels, calibrated_probs)
    print(f"ECE for Platt-Calibrated Model: {ece_calibrated:.3f}")


    # Save results for later analysis and alignment
    torch.save(calibration_results, "../../../tudelft.net/staff-umbrella/EleniSalient/calibration_results.pth")
    print("Calibration results saved.")
# ----------------------------------------------------------------------------------------------------------------------------

#     # Try calibration model from github
#     scaled_model = ModelWithTemperature(model)
#     scaled_model.set_temperature(calibration_loader)
#     scaled_model.eval()

#     # (Optional) Evaluate on the train set for analysis
#     print("Evaluating on the train set for analysis")
#     train_loss, train_results, _, _  = evaluate_model(scaled_model, train_loader, criterion)
#     print(f"Train Set Loss: {train_loss}")


#     # Initialize results dictionary
#     calibration_results = {
#         "patient_ids": [],
#         "segment_indices": [],
#         "predictions": [],
#         "true_labels": [],
#         "calibrated_probs": []
#     }

#     all_probs = []
#     all_patient_numbers = []
#     all_segment_orders = []
#     with torch.no_grad():   
#         for inputs, labels , patient_numbers,segments_order in train_loader:
#             logits = model(inputs)  # Scaled logits
#             probs = torch.softmax(logits, dim=1)  # Calibrated probabilities
#             predictions = (probs[:, 1] > 0.5).int()  # Binary predictions for class 1

#             # Save results
#             calibration_results["patient_ids"].extend(patient_numbers.tolist())
#             calibration_results["segment_indices"].extend(segments_order.tolist())
#             calibration_results["predictions"].extend(predictions.tolist())
#             calibration_results["true_labels"].extend(labels.tolist())
#             calibration_results["calibrated_probs"].extend(probs.tolist())  # Store entire probability distribution

#     # # Save results for later analysis and alignment
#     # torch.save(calibration_results, "../../../tudelft.net/staff-umbrella/EleniSalient/calibration_results.pth")
#     # print("Calibration results saved.")

#     # EVALUATION OF CALIBRATION
#     # Extract probabilities for class 1 and true labels
#     calibrated_probs = torch.tensor(calibration_results["calibrated_probs"])[:, 1].numpy()  # Probabilities for class 1
#     true_labels = torch.tensor(calibration_results["true_labels"]).numpy()  # True labels

#     # Compute calibration curve
#     fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, calibrated_probs, n_bins=10)
# # 0.25081757776902447 // ECE for calibrated model: 0.22317193892992856
#     # Plot reliability diagram
#     plt.figure(figsize=(8, 6))
#     plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated Model")
#     plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
#     plt.xlabel("Mean Predicted Value")
#     plt.ylabel("Fraction of Positives")
#     plt.title("Reliability Diagram")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/calibration_curve.png')
#     plt.close()

#     # Compute ECE for calibrated probabilities
#     ece_calibrated = expected_calibration_error(true_labels, calibrated_probs)
#     print(f"ECE for calibrated model: {ece_calibrated}")

#     plt.figure(figsize=(8, 6))
#     plt.hist(calibrated_probs, bins=20, alpha=0.7, label="Calibrated Probabilities")
#     plt.xlabel("Predicted Probability")
#     plt.ylabel("Frequency")
#     plt.title("Probability Distribution After Calibration")
#     plt.legend()
#     plt.savefig('../../../tudelft.net/staff-umbrella/EleniSalient/probability_distribution.png')
#     plt.close()