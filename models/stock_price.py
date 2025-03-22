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





