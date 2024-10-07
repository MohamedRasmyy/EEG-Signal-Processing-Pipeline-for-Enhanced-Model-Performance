#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# Load the preprocessed data
X = np.load('X_all_ICA_ASR.npy')
Y = np.load('Y_all_ICA_ASR.npy')


# In[4]:


import random
import torch
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# In[5]:


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=.1,random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)  

X_train_flatten = X_train.reshape(-1, X_train.shape[-1])  # Shape: (num_train_epochs * num_channels, seq_len)
X_val_flatten = X_val.reshape(-1, X_val.shape[-1])        # Validation set
X_test_flatten = X_test.reshape(-1, X_test.shape[-1])  # Shape: (num_test_epochs * num_channels, seq_len)

mean = X_train_flatten.mean(axis=0)  # Compute the mean for each feature
X_train_centered = X_train_flatten - mean
X_val_centered = X_val_flatten - mean  # Center validation data using train set mean
X_test_centered = X_test_flatten - mean  # Apply the same mean to X_test

scaler = StandardScaler(with_mean=False)  # Disable mean subtraction since we already did it
X_train_scaled = scaler.fit_transform(X_train_centered)  # Fit on X_train
X_val_scaled = scaler.transform(X_val_centered)  # Apply the same scaler to validation data
X_test_scaled = scaler.transform(X_test_centered)  # Transform X_test using the same scaler

X_train_final = X_train_scaled.reshape(X_train.shape)  # Shape: (num_train_epochs, num_channels, seq_len)
X_val_final = X_val_scaled.reshape(X_val.shape)        # Shape: (num_val_epochs, num_channels, seq_len)
X_test_final = X_test_scaled.reshape(X_test.shape)  # Shape: (num_test_epochs, num_channels, seq_len)

print("Train set shape:", X_train_final.shape)
print("Validation set shape:", X_val_final.shape)
print("Test set shape:", X_test_final.shape)


# In[6]:


# 1. Convert Numpy Arrays to PyTorch Tensors
X_train1=torch.tensor(X_train_final,dtype=torch.float32)
X_val1=torch.tensor(X_val_final,dtype=torch.float32)
X_test1=torch.tensor(X_test_final,dtype=torch.float32)

y_train1=torch.tensor(y_train,dtype=torch.long)
y_val1=torch.tensor(y_val,dtype=torch.long)
y_test1=torch.tensor(y_test,dtype=torch.long)

# 2. Create TensorDatasets for Train, Validation, and Test Sets
train_dataset=TensorDataset(X_train1,y_train1)
val_dataset=TensorDataset(X_val1,y_val1)
test_dataset=TensorDataset(X_test1,y_test1)

# 3. Create DataLoaders for Batch Processing
train_loader=DataLoader(train_dataset,batch_size=100,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=100,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=100,shuffle=False)


# In[7]:


import torch.nn as nn
import torch.optim as optim

# Define the CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, input_channels, output_channel, lstm_hidden_dim1, lstm_hidden_dim2, output_channel2, num_classes):
        super(CNNLSTM, self).__init__()
        
        #First CNN Layer
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, output_channel, padding=0, kernel_size=20, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(output_channel),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5)
        )
        
        #First LSTM layers with bidirectional
        self.lstm1 = nn.LSTM(output_channel, hidden_size=lstm_hidden_dim1, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)
        
        #Second CNN Layer
        self.cnn2 = nn.Sequential(
            nn.Conv1d(lstm_hidden_dim1*2, output_channel2, padding=0, kernel_size=10, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(output_channel2),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.5)
        )
        
        #Second Lstm layer with bidirectional
        self.lstm2 = nn.LSTM(output_channel2, lstm_hidden_dim2, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_dim2 * 2 * 3, num_classes)
        self._initialize_weights()

    def forward(self, x):
        
        x = self.cnn(x)  # Apply CNN
        x = x.permute(0, 2, 1)  # Permute for LSTM (batch_size, seq_len, num_features)
        
        x, _ = self.lstm1(x)  # First LSTM layer
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 1)  # Permute back for CNN
        x = self.cnn2(x)  # Apply second CNN layer
        
        x = x.permute(0, 2, 1)  # Permute for LSTM
        
        x, _ = self.lstm2(x) # Second LSTM layer
        x = self.dropout2(x)

        x = x.contiguous().view(x.size(0), -1)  # Flatten for FC layer
        x = self.fc(x)  # Apply fully connected layer
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        n = param.size(0)
                        param[n // 4:n // 2].data.fill_(1.0)  # Set forget gate bias to 1.0

input_channels = 22  
output_channel = 40
output_channel2 = 30
lstm_hidden_dim1 = 70
lstm_hidden_dim2 = 50
num_classes = 4  

# Instantiate the model
model = CNNLSTM(input_channels, output_channel, lstm_hidden_dim1, lstm_hidden_dim2, output_channel2, num_classes)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=.002)
# Training the model
num_epochs = 250
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training phase
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    val_loss = running_val_loss / len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
# Testing Phase
model.eval()
correct = 0
total = 0
with torch.no_grad():	
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_accuracy = 100 * correct / total
print(f'Final accuracy on test set: {final_accuracy:.2f}%') 

