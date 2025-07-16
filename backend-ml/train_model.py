import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # for saving scaler

# 1. Load and prepare data
df = pd.read_csv("ckd.csv")
df = df[['age', 'bp', 'sg', 'al', 'classification']]

# Map classification labels to binary
df['classification'] = df['classification'].str.strip().map({'ckd': 1, 'notckd': 0})

# Drop rows with missing data
df.dropna(inplace=True)

X = df[['age', 'bp', 'sg', 'al']].values.astype(np.float32)
y = df['classification'].values.astype(np.float32).reshape(-1, 1)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 5. Define model architecture
class CKDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = CKDModel()

# 6. Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 8. Save model weights and scaler
torch.save(model.state_dict(), "ckd_model.pt")
joblib.dump(scaler, "scaler.pkl")

print("Training complete. Model and scaler saved.")