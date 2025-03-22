import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

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
def create_sequences(data, seq_length, forecast_horizon):
    """
    Creates sequences for seq2seq training.

    Parameters:
    - data: Scaled numpy array of shape (num_samples, num_features)
    - seq_length: Lookback window (e.g., 60 days)"
    - forecast_horizon: Number of future days to predict (e.g., 180 days)

    Returns:
    - X: Array of input sequences with shape (num_sequences, seq_length, num_features)
    - y: Array of target sequences (future closing prices) with shape (num_sequences, forecast_horizon)
    
    """
    X, y = [], []
    #3 is the index for the closing price
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        # we are going with 60 day sequence of all features
        X.append(data[i: i + seq_length])
        #Target  as the  sequence of future closing price
        y.append(data[i + seq_length: i + seq_length + forecast_horizon, 3])

    return np.array(X), np.array(y)

SEQ_LENGTH = 60
FORECAST_HORIZON = 180 
X_sequences, y_sequence = create_sequences(data_scaled, SEQ_LENGTH, FORECAST_HORIZON)
 
print("Shape of input sequences (X):", X_sequences.shape)
print("Shape of targets (y):", y_sequence.shape)


#Splitting the data into train, validation, and test sets
total_samples = len(X_sequences)
train_size = int(0.7 * total_samples)
val_size = int(0.15 * total_samples)
test_size = total_samples - train_size - val_size

print("Training samples:", train_size)
print("Validation samples:", val_size)
print("Testing samples:", test_size)

#Splitting the data
X_train = X_sequences[:train_size]
y_train = y_sequence[:train_size]
X_val = X_sequences[train_size:train_size + val_size]
y_val = y_sequence[train_size:train_size + val_size]
X_test = X_sequences[train_size + val_size:]
y_test = y_sequence[train_size + val_size:]

#converting y to have an extra dimension
y_train = np.expand_dims(y_train, axis=2)
y_val = np.expand_dims(y_val, axis=2)
y_test = np.expand_dims(y_test, axis=2)

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


#Building the lstm model

#Processes the input sequence
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        # Here, input size is 1 because we feed one closing price at each step.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden, cell):
        # x: (batch_size, 1, 1)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)  # (batch_size, 1, 1)
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, forecast_horizon):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.forecast_horizon = forecast_horizon
        
    def forward(self, src, teacher_forcing_ratio=0.5, target=None):
        """
        src: Input tensor (batch_size, seq_length, input_size)
        teacher_forcing_ratio: probability of using the true target as the next input
        target: Ground truth tensor (batch_size, forecast_horizon, 1)
        """
        batch_size = src.size(0)
        hidden, cell = self.encoder(src)
        
        #  hold predictions
        outputs = torch.zeros(batch_size, self.forecast_horizon, 1).to(src.device)
        
        # Initialize the first input to the decoder as the last closing price from the input sequence
        decoder_input = src[:, -1, 3].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
        
        for t in range(self.forecast_horizon):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = decoder_output.squeeze(1)
            
            # teacher forcing
            if target is not None and np.random.rand() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)  # (batch_size, 1, 1)
            else:
                decoder_input = decoder_output  # Use the prediction as the next input
                
        return outputs




encoder = Encoder(input_size=5, hidden_size=64, num_layers=2, dropout=0.2)
decoder = Decoder(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
model = Seq2Seq(encoder, decoder, forecast_horizon=FORECAST_HORIZON)
#loss function and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#now we train
num_epochs = 150
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_val_tensor   = X_val_tensor.to(device)
y_val_tensor   = y_val_tensor.to(device)

best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
     # Forward pass with teacher forcing during training
    train_output = model(X_train_tensor, teacher_forcing_ratio=0.5, target=y_train_tensor)
    train_loss = loss(train_output, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor, teacher_forcing_ratio=0.0)
        val_loss = loss(val_output, y_val_tensor)
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), 'best_seq2seqmodel.pth')
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

#evaluating on the test set
model.load_state_dict(torch.load('best_seq2seqmodel.pth'))
model.eval()
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)
with torch.no_grad():
    test_pred = model(X_test_tensor, teacher_forcing_ratio=0.0)
    test_loss = loss(test_pred, y_test_tensor)
print(f"Test Loss: {test_loss.item():.4f}")



#change the data to normal because right now we scalared it
#all we have tro graoh actual labels vs predicted labels
#and if it looks bad, i will take over, if it doesnt graph, create graohs 6 months, trends.