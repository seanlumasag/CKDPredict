# Import required libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the scaler

# Load dataset from CSV file
df = pd.read_csv("ckd.csv")

# Select relevant columns for model input and output
df = df[['age', 'bp', 'sg', 'al', 'classification']]

# Clean and encode the target variable: 'ckd' -> 1, 'notckd' -> 0
df['classification'] = df['classification'].str.strip().map({'ckd': 1, 'notckd': 0})

# Drop rows with missing values
df.dropna(inplace=True)

# Separate features and target variable
X = df[['age', 'bp', 'sg', 'al']].values.astype(np.float32)  # Features
y = df['classification'].values.astype(np.float32).reshape(-1, 1)  # Target

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features using standard scaling (mean=0, std=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert NumPy arrays to PyTorch tensors for model training
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define a neural network for CKD prediction
class CKDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),  # Input layer with 4 features → 16 neurons
            nn.ReLU(),         # Activation function
            nn.Linear(16, 8),  # Hidden layer: 16 → 8 neurons
            nn.ReLU(),         # Activation function
            nn.Linear(8, 1),   # Output layer: 8 → 1 neuron (binary prediction)
            nn.Sigmoid()       # Sigmoid for probability output
        )

    # Define forward pass
    def forward(self, x):
        return self.net(x)

# Initialize model
model = CKDModel()

# Define binary cross-entropy loss (used for binary classification)
criterion = nn.BCELoss()

# Use Adam optimizer to update model weights
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop for 100 epochs
epochs = 100
for epoch in range(epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear previous gradients
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model weights

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save trained model weights to file
torch.save(model.state_dict(), "ckd_model.pt")

# Save the scaler to apply same transformation during prediction
joblib.dump(scaler, "scaler.pkl")

print("Training complete. Model and scaler saved.")
