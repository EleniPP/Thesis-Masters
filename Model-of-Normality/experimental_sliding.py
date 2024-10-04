import torch
import pickle
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import csv
from temperature_scaling import ModelWithTemperature
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from torch.optim.lr_scheduler import StepLR
import h5py
import gc
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.functions import udf
import psutil

# Function to check available memory
def available_memory():
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 2)  # Return available memory in MB

def load_features_from_hdf5(hdf5_file_path):
    patient_features = []
    
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        for patient_id in hdf5_file:
            # Load each patient's features into a NumPy array
            patient_data = hdf5_file[patient_id][:]
            print(f"Loaded features for {patient_id} with shape {patient_data.shape}")
            patient_features.append(patient_data)
    
    # Convert the list of features to a NumPy array with dtype=object
    feature_patients = np.array(patient_features, dtype=object)
    return feature_patients

# Initialize Spark
spark = SparkSession.builder.appName("FeatureProcessing") .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g")\
    .config("spark.memory.fraction", "0.8").config("spark.storage.memoryFraction", "0.1")\
    .getOrCreate()
# Example usage to load the features
audio_features = load_features_from_hdf5('D:/Data/log_mel_sliding/audio_features2_sliding.h5')
print(f"Loaded audio features for all patients. Shape: {audio_features.shape}")

visual_features = load_features_from_hdf5('D:/Data/aggr_visual_sliding/visual_features_sliding2.h5')
print(f"Loaded visual features for all patients. Shape: {visual_features.shape}")

print(audio_features.shape)
print(audio_features[0].shape)

print(visual_features.shape)
print(visual_features[0].shape)

print('1')

with open('D:/Data/labels.json', 'r') as file:
    labels_dict = json.load(file)
del labels_dict['492']

print('2')

numbers = [int(key) for key in labels_dict.keys()]

print('3')

labels = list(labels_dict.values())
tlabels = torch.tensor(labels)
print('4')
# Convert audio and visual features to torch tensors 
audio_features = [torch.from_numpy(a) for a in audio_features]
print('5')
visual_features = [torch.from_numpy(v) for v in visual_features]
print('6')
gc.collect()
# Check available memory
print(f"Available memory before processing: {available_memory()} MB")

# Create the RDD from the numpy array (do this in smaller chunks if needed)
batch_size = 100  # Adjust based on your memory capacity
for i in range(0, len(audio_features), batch_size):
    audio_batch = audio_features[i:i + batch_size]
    visual_batch = visual_features[i:i + batch_size]
    
    # Create RDD for the batch
    audio_rdd = spark.sparkContext.parallelize([(i.tolist(), idx) for idx, i in enumerate(audio_batch, i)])
    visual_rdd = spark.sparkContext.parallelize([(i.tolist(), idx) for idx, i in enumerate(visual_batch, i)])

    # Convert to DataFrames
    audio_features_df = audio_rdd.toDF(["audio", "id"])
    visual_features_df = visual_rdd.toDF(["visual", "id"])

    # Proceed with the join and concatenation here for each batch
    multimodal_df = audio_features_df.join(visual_features_df, on="id")
    print(f"Processed batch {i//batch_size + 1}")
# audio_rdd = spark.sparkContext.parallelize([(i.tolist(), idx) for idx, i in enumerate(audio_features)], numSlices=10)
# visual_rdd = spark.sparkContext.parallelize([(i.tolist(), idx) for idx, i in enumerate(visual_features)], numSlices=10)
# Create DataFrames directly from the NumPy arrays and add an `id` column using range
# audio_features_df = spark.createDataFrame([(i.tolist(), idx) for idx, i in enumerate(audio_features)], ["audio", "id"])
# visual_features_df = spark.createDataFrame([(i.tolist(), idx) for idx, i in enumerate(visual_features)], ["visual", "id"])
# audio_features_df = audio_rdd.toDF(["audio", "id"])
# visual_features_df = visual_rdd.toDF(["visual", "id"])
print(f"Available memory after processing: {available_memory()} MB")
print(1/0)
multimodal_df = audio_features_df.join(visual_features_df, on="id")

# Define a UDF to concatenate the audio and visual features along the last dimension
concat_udf = udf(lambda a, v: a + v, ArrayType(FloatType()))

# Apply the UDF to create a combined "multimodal" column by concatenating audio and visual features
multimodal_df = multimodal_df.withColumn("multimodal", concat_udf("audio", "visual"))

# Collect the multimodal features as a list of NumPy arrays
multimodal_features = multimodal_df.select("multimodal").rdd.map(lambda row: np.array(row[0])).collect()
print('hi')
print(f"Shape of first patient's multimodal features: {multimodal_features[0].shape}")
# batch_size = 10 # Adjust this based on your available memory
# multimodal_features = []

# for i in range(0, len(audio_features), batch_size):
#     batch_audio = audio_features[i:i+batch_size]
#     batch_visual = visual_features[i:i+batch_size]
#     batch_multimodal = [torch.cat((a, v), dim=1) for a, v in zip(batch_audio, batch_visual)]
#     del batch_audio
#     del batch_visual
#     multimodal_features.extend(batch_multimodal)
#     print(f"Processed batch {i//batch_size + 1}")
# multimodal_features = [torch.cat((a, v), dim=1) for a, v in zip(audio_features, visual_features)]
# print(multimodal_features.shape)
# Normalize each tensor individually
normalized_multimodal = []
for tensor in multimodal_features:
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    normalized_tensor = (tensor - mean) / (std + 1e-5)
    normalized_multimodal.append(normalized_tensor)

# Identify and remove NaN entries and flatten inputs for trainig
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


def flatten(multimodal_features, labels):
    flattened_features = []
    flattened_labels = []
    num_seg_per_patient = []
    segment_indices = []  # To track original chronological order within each patient
    for i, (features, label) in enumerate(zip(multimodal_features, labels)):
        cnt = 0
        for idx,segment in enumerate(features):
            cnt += 1
            flattened_features.append(segment)
            flattened_labels.append(label)
            segment_indices.append(idx)
        num_seg_per_patient.append(cnt)
    # Convert flattened lists to tensors
    flattened_features = torch.stack(flattened_features)
    flattened_labels = torch.tensor(flattened_labels)
    segment_per_patient = torch.tensor(num_seg_per_patient)
    segment_indices = torch.tensor(segment_indices) #so this contains the index of each segment for all the segments in the train multimodal
    return flattened_features,flattened_labels,segment_per_patient, segment_indices

def get_split(split):
    base_path = "D:/Data/"
    extension = "_split.csv"
    file = f"{base_path}{split}{extension}"
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
    threshold = 0.47
    with torch.no_grad():  # Disable gradient calculation
        for features, labels in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
            loss = criterion(outputs, labels)
            # loss = xon(outputs.view(-1, 2), labels.view(-1))
            test_loss += loss.item()

            # Calculate accuracy for this batch
            # Convert logits to probabilities and predictions
            probs = torch.softmax(outputs, dim=1)[:,1]  # Probability of class 1 
            # _, predicted = torch.max(outputs, -1)
            predicted = (probs > threshold).int()
            # Append predictions and labels for later evaluation
            all_predictions.append(predicted)  # Move to CPU if using GPU
            all_labels.append(labels)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Concatenate all predictions and labels from all batches
    all_predictions = torch.cat(all_predictions).numpy()
    all_labels = torch.cat(all_labels).numpy()
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    print(classification_report(all_labels, all_predictions))
    return test_loss / len(dataloader),accuracy, all_predictions, all_labels

def validate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for features, labels in dataloader:
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue
            outputs = model(features)
            loss = criterion(outputs.view(-1, 2), labels.view(-1))
            val_loss += loss.item()
    return val_loss / len(dataloader)

# Function to filter features and labels based on patient numbers
def filter_by_patient_numbers(features, labels, patient_numbers):
    indices = [patient_to_index[patient] for patient in patient_numbers if patient in patient_to_index]
    filtered_features = [features[idx] for idx in indices]
    filtered_labels = [labels[idx] for idx in indices]
    return filtered_features, filtered_labels


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights, should be a tensor with shape [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # No reduction to compute per-sample loss
        
        # Get the predicted probabilities for the correct class (pt)
        pt = torch.exp(-ce_loss)  # Probabilities of the true class

        # If alpha (class weights) are provided, apply them
        if self.alpha is not None:
            # Ensure alpha is properly broadcasted (shape [batch_size])
            at = self.alpha.gather(0, targets)  # Gather correct alpha for each target in the batch
            ce_loss = at * ce_loss  # Apply class weights to the cross-entropy loss

        # Apply focal loss scaling
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply the reduction (mean, sum, etc.)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # Return raw losses if no reduction is needed

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
        self.segment_indices = segment_indices  # Original segment order

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.patient_numbers[idx], self.segment_indices[idx]

# Function to collate and pad sequences within a batch
def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignore index in loss
    return data_padded, labels_padded

class DepressionPredictor1(nn.Module):
    def __init__(self):
        super(DepressionPredictor1, self).__init__()
        # Completely simplified:
        # self.classifier = nn.Sequential(
        #     nn.Linear(3072, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 2)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(3072, 1024),
            # nn.Linear(3072, 512),
            # nn.BatchNorm1d(1024),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.6),  # Assuming some dropout for regularization
            nn.Linear(1024, 256),
            # nn.Linear(512, 128),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.6),  # Dropout to prevent overfitting
            nn.Linear(256, 2) #2 for softmax and 1 for sigmoid
            # nn.Linear(128, 2)
        )
# possibleTODO - batch size change?
    def forward(self, x):
        x = self.classifier(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Found nan or inf in model output")
        # _, num_segments, _ = x.shape
        # x = x.view(-1, num_segments, 2)  # Reshape to [batch_size, segments, classes]
        return x  # Return logits for calibration and softmax application externally
    

def plot_losses(train_losses,val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()


def train_final_model(model, dataloader,optimizer, criterion, epochs = 10):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0
        for i, (features, labels,_,_) in enumerate(dataloader):
            optimizer.zero_grad()
            # DEBUG: Ensure features do not contain NaN or Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            outputs = model(features)  # Outputs are logits for two classes
            loss = criterion(outputs, labels)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
        # Logging the average training loss for this epoch
        avg_train_loss = total_loss / len(dataloader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')

    # Save the final trained model
    torch.save(model.state_dict(), 'final_model1.pth')
    print("Final model saved as 'final_model1.pth'")

    # Optionally, plot the training losses
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    return model

def train_model(model, dataloader,val_loader, optimizer, scheduler, criterion,epochs=30):
    # Early stopping
    early_stopping_patience = 5  # Number of epochs with no improvement after which training will be stopped
    best_val_loss = float('inf')
    patience_counter = 0
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        total_loss = 0
        epoch_probabilities = []
        for i, (features,labels) in enumerate(dataloader):
            optimizer.zero_grad()
            # DEBUG: Ensure features do not contain NaN or Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            outputs = model(features)  # Outputs are now probabilities for two classes for each segment
            # loss = criterion(outputs.view(-1, 2), labels[i].view(-1))  # Reshape appropriately if needed
            loss = criterion(outputs,labels)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # retain_graph=True
            optimizer.step()
            total_loss += loss.item()    
            # store probabilities for all the patients
        
        # TODO: me poio val kano edo validate? poios val_loader einai aftos?
        val_loss = validate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model if it's the best one yet
            torch.save(model.state_dict(), 'best_model1.pth')
        else:
            patience_counter += 1
    
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
        # print(f'Epoch {epoch+1}, Loss: {total_loss}')
        train_losses.append(total_loss/len(dataloader))
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(dataloader)}, Validation Loss: {val_loss}')
        # Step the scheduler
        # scheduler.step(val_loss)
    plot_losses(train_losses,val_losses)
    # all_probabilities = np.array(all_probabilities, dtype=object)
    # np.save('V:/staff-umbrella/EleniSalient/Data/probability_distributions.npy', all_probabilities)
    return model

# Use the temp labels since we only have 2 patients

# labels = [torch.full((features.shape[0],), tlabels[i], dtype=torch.long) for i, features in enumerate(normalized_multimodal)]

# Instantiate the model
model = DepressionPredictor1()

# # Define the optimizer
# # optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=2e-4)
# # optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# # 5e-3
# # Define the learning rate scheduler
# # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3, verbose=True)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
# Calibrate the model

# Get splits
train_split = get_split('train')
test_split = get_split('test')
val_split = get_split('dev')
train_split = train_split[:-1] #because i didnt put the 491 in the labels_dict and in the data

del numbers[117]
del numbers[99]
# Create a dictionary mapping patient numbers to their corresponding index
patient_to_index = {patient_num: idx for idx, patient_num in enumerate(numbers)}
for i, features in enumerate(normalized_multimodal):
    if torch.isnan(features).any():
        print(f"NaN values in normalized_multimodal before dataloaders at index {i}")

# Filter the features and labels for each split
train_multimodal, train_labels = filter_by_patient_numbers(normalized_multimodal, labels, train_split)
train_multimodal,train_labels,segments_per_patient_train,segments_order_train = flatten(train_multimodal,train_labels)

# array that has the patient number for each segment in the train multimnodal
segments_patients_train = [num for count, num in zip(segments_per_patient_train, train_split) for _ in range(count)]

#Check inbalance
unique, counts = np.unique(train_labels, return_counts=True)
class_distribution = dict(zip(unique, counts))


val_multimodal, val_labels = filter_by_patient_numbers(normalized_multimodal, labels, val_split)
val_multimodal,val_labels,segments_per_patient_val,segments_order_val = flatten(val_multimodal,val_labels)
segments_patients_val = [num for count, num in zip(segments_per_patient_val, val_split) for _ in range(count)]
# print(val_multimodal.shape)
# print(val_multimodal[0].shape)
# Leave the test alone for now :)
test_multimodal, test_labels = filter_by_patient_numbers(normalized_multimodal, labels, test_split)
test_multimodal,test_labels,segments_per_patient_test,segments_order_test = flatten(test_multimodal,test_labels)
segments_patients_test = [num for count, num in zip(segments_per_patient_test, test_split) for _ in range(count)]
# print(test_multimodal.shape)
# print(test_multimodal[0].shape)
# Create training, validation, and test datasets and dataloaders
train_dataset = DepressionDataset(train_multimodal, train_labels,segments_patients_train,segments_order_train)
val_dataset = DepressionDataset(val_multimodal, val_labels,segments_patients_val,segments_order_val)
test_dataset = DepressionDataset(test_multimodal, test_labels,segments_patients_test,segments_order_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# Do i need to shuffle the validation set as well?
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
# Leave the test alone
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
# criterion = nn.CrossEntropyLoss(ignore_index=-100)

# probability_distribution = train_model(model, train_loader,val_loader, optimizer, criterion)


# -----------CROSS VALIDATION-------------------------------------
# Number of folds for cross-validation
n_splits = 5 #for the debugging of the temperature scaling

# Initialize KFold
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Assuming train_multimodal and train_labels are already in the shape (number_of_segments x features)
# and labels respectively
features = train_multimodal  # Use your multimodal features directly
labels = train_labels  # Use your labels directly
# Store results for each fold
fold_results = []

# Cross-Validation Loop
for fold, (train_index, val_index) in enumerate(kf.split(features)):
    print(f"Fold {fold + 1}/{n_splits}")
    
    # Split data into training and validation based on indices
    train_features, val_features = features[train_index], features[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]


    # Assuming y contains your labels (for the entire dataset)
    unique, counts = np.unique(train_labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    print(class_distribution)
    
    # Initialize SMOTE
    smote = SMOTE(sampling_strategy=0.5)
    # Assuming you have your training data in X_train and y_train
    # adasyn = ADASYN(sampling_strategy='auto', random_state=42, n_neighbors=2)
    # Try to solve the imbalance problem with oversampling
    # train_features, train_labels = smote.fit_resample(train_features, train_labels)
    
    # Create DataLoaders for this fold
    train_dataset = DepressionDatasetCross(train_features, train_labels)
    val_dataset = DepressionDatasetCross(val_features, val_labels)
    
    training_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Initialize a new model for this fold
    model = DepressionPredictor1()
    
    # Define the optimizer and learning rate scheduler
    # optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Adam default learning rate is 0.001
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    # Initialize the scheduler (e.g., StepLR)
    # scheduler = StepLR(optimizer, step_size=8, gamma=0.7)
    
    # Define the loss function
    # criterion = nn.CrossEntropyLoss(ignore_index=-100,weight=torch.tensor([1.0, 1.2]))
    alpha = torch.tensor([1.0, 1.2])  # Weights for class 0 and class 
    # when it goes up like 1.3 from 1.2 then precision is higher for class 1 but recall is lower
    criterion = FocalLoss(alpha=alpha, gamma=1.6)
    
    # Train the model for this fold
    probability_distribution = train_model(model, training_loader,valid_loader, optimizer,scheduler, criterion)
    
    # Evaluate on the validation set
    val_loss, accuracy, _, _ = evaluate_model(model, valid_loader, criterion)
    
    # Store results for this fold
    fold_results.append({
        'fold': fold + 1,
        'val_loss': val_loss,
        'accuracy': accuracy
    })
    
    print(f"Validation Loss for fold {fold + 1}: {val_loss:.4f}")
    print(f"Validation Accuracy for fold {fold + 1}: {accuracy:.4f}")

# Calculate average validation loss and accuracy across all folds
avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
avg_accuracy = np.mean([result['accuracy'] for result in fold_results])

print(f"Average Validation Loss across {n_splits} folds: {avg_val_loss:.4f}")
print(f"Average Validation Accuracy across {n_splits} folds: {avg_accuracy:.4f}")


# # Final training of the model in the whole training set
# # Assuming you have your full training data in train_loader
# final_model = DepressionPredictor1()  # Initialize your model architecture

# # Define optimizer and criterion
# optimizer = optim.Adam(final_model.parameters(), lr=1e-5, weight_decay=1e-4)
# criterion = nn.CrossEntropyLoss()

# # Train the model on the full dataset
# final_model = train_final_model(final_model, train_loader, optimizer, criterion, epochs=8)


# # Load the saved model
# model = DepressionPredictor1()  # Initialize your model architecture
# model.load_state_dict(torch.load('final_model1.pth'))
# model.eval()  # Set the model to evaluation mode
# print("Model loaded and ready for calibration")
# # Try calibration model from github
# scaled_model = ModelWithTemperature(model)
# scaled_model.set_temperature(val_loader)

# scaled_model.eval()
# all_probs = []
# all_patient_numbers = []
# all_segment_orders = []
# with torch.no_grad():   
#     for inputs, _ , patient_numbers,segments_order in train_loader:
#         logits = scaled_model(inputs)  # Scaled logits
#         probs = torch.softmax(logits, dim=1)  # Calibrated probabilities
#         all_probs.append(probs)
#         # Save patient numbers from the current batch (same order as the probabilities predicted for the same segments)
#         all_patient_numbers.extend(patient_numbers.tolist())
#         all_segment_orders.extend(segments_order.tolist())
# # Combine probabilities from all batches
# all_probs = torch.cat(all_probs, dim=0)
# # Convert patient numbers list to a tensor
# all_patient_numbers = torch.tensor(all_patient_numbers)
# all_segment_orders = torch.tensor(all_segment_orders)

# torch.save(all_probs, 'probability_distributions.pth')
# torch.save(all_patient_numbers, 'all_patient_numbers.pth')
# torch.save(all_segment_orders, 'all_segment_orders.pth')
 