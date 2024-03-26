import numpy as np
import torch
import torch.nn as nn

visual = np.load('C:/Users/eleni/Data/visual.npy')
tvisual = torch.from_numpy(visual)

input_data = tvisual.unsqueeze(1)  # Add channel dimension

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=7)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.05)
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=7)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7)
        
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten the output before passing it to fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x

# Example usage:
# Assuming input size is 20 (features extracted from AUs) and number of classes is 10
model = CNN(input_size=20, num_classes=10)
print(model)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model (example: assuming binary classification)

#TODO Replace 'labels' with MY LABELS
labels = torch.randint(0, 2, (input_data.shape[0],), dtype=torch.float32)
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')