import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import pickle

#Loading the csv file
df = pd.read_csv("costco_stock_data.csv")

#converting the date column to a datetime type
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)#sorting by date

#checking for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

#Feature Selection
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features].copy()


#Scaling the data to transform all features to the range[0.1].
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

#save the scaler value for later
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nData has been scaled. Scaled data shape:", data_scaled.shape)

#create sequence for lstm
def create_sequences(data, seq_length):
    """
    Creates sequences for LSTM training.

    Parameters:
    - data: Scaled numpy array of shape (num_samples, num_features)
    - seq_length: Lookback window (e.g., 60 days)"

    Returns:
    - X: Array of input sequences with shape (num_sequences, seq_length, num_features)
    - y: Array of targets (next day 'Close') with shape (num_sequences,)
    
    """
    X, y = [], []
    #3 is the index for the closing price
    for i in range(len(data) - seq_length):
        # we are going with 60 day sequence of all features
        X.append(data[i: i + seq_length])
        #Target is next day's closing price
        y.append(data[i + seq_length, 3])

    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X_sequences, y_targets = create_sequences(data_scaled, SEQ_LENGTH)

print("Shape of input sequences (X):", X_sequences.shape)
print("Shape of targets (y):", y_targets.shape)


#Splitting the data into train, validation, and test sets

total_samples = len(X_sequences)
train_size = int(0.7 * total_samples)
val_size = int(0.15 * total_samples)
test_size = total_samples - train_size - val_size

print("Training samples:", train_size)
print("Validation samples:", val_size)
print("Testing samples:", test_size)

#Training set
X_train = X_sequences[:train_size]
y_train = y_targets[:train_size]

#Validation set
X_val = X_sequences[train_size:train_size + val_size]
y_val = y_targets[train_size:train_size + val_size]

#Test set
X_test = X_sequences[train_size + val_size:]
y_test = y_targets[train_size + val_size:]

#Converting the data into pytorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

print("\nPyTorch Tensor Shapes:")
print("X_train_tensor:", X_train_tensor.shape)  # (522, 60, 5)
print("y_train_tensor:", y_train_tensor.shape)  # (522,)
print("X_val_tensor:", X_val_tensor.shape)      # (112, 60, 5)
print("y_val_tensor:", y_val_tensor.shape)      # (112,)
print("X_test_tensor:", X_test_tensor.shape)    # (113, 60, 5)
print("y_test_tensor:", y_test_tensor.shape)    # (113,)








    

     











