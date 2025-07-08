import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and prepare data
df = pd.read_csv("ckd.csv")
df = df[['age', 'bp', 'sg', 'al', 'classification']]
df['classification'] = df['classification'].str.strip().map({'ckd': 1, 'notckd': 0})
df.dropna(inplace=True)

# Features and target
X = df[['age', 'bp', 'sg', 'al']].values.astype(np.float32)
y = df['classification'].values.astype(np.float32).reshape(-1, 1)

# Normalize inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

# Define simple linear model
model = nn.Sequential(
    nn.Linear(4, 1),   # 4 input features → 1 output
    nn.Sigmoid()       # Use sigmoid for output between 0 and 1
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Save model and scaler
torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler}, 'ckd_model.pt')

# Sample test data (modifiable)
sample_data = np.array([
    [20, 80, 1.020, 1],
    [45, 70, 1.015, 2],
    [60, 90, 1.010, 3],
    [30, 85, 1.025, 0],
    [50, 95, 1.005, 4]
], dtype=np.float32)

# Normalize sample using same scaler
sample_data = scaler.transform(sample_data)
sample_tensor = torch.tensor(sample_data)

# Predict
with torch.no_grad():
    preds = model(sample_tensor)
    binary_preds = (preds >= 0.5).int()
    print("Predictions:", binary_preds.view(-1).tolist())
