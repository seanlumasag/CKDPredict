import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# -----------------------------
# 1. Load and Preprocess Data
# -----------------------------
df = pd.read_csv("ckd.csv")
df = df[['age', 'bp', 'sg', 'al', 'classification']]
df.dropna(subset=['age', 'bp', 'sg', 'al', 'classification'], inplace=True)

df['classification'] = df['classification'].str.strip().str.lower()
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})
df.dropna(subset=['classification'], inplace=True)

X = df[['age', 'bp', 'sg', 'al']].values.astype('float32')
y = df['classification'].values.astype('float32')

# Scale features using StandardScaler (important!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['age', 'bp', 'sg', 'al']])



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test).unsqueeze(1)

# -----------------------------
# 2. Define Model
# -----------------------------
class CKDModel(nn.Module):
    def __init__(self):
        super(CKDModel, self).__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = CKDModel()

# -----------------------------
# 3. Train Model
# -----------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

for epoch in range(1000):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -----------------------------
# 4. Evaluate Model
# -----------------------------
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (predicted_classes.eq(y_test_tensor).sum() / y_test_tensor.size(0)).item()
    print(f"\nTest Accuracy: {accuracy:.2f}")

# -----------------------------
# 5. Save Model and Scaler
# -----------------------------
torch.save(model.state_dict(), "ckd_model.pt")
print("Model saved as 'ckd_model.pt'")

# -----------------------------
# 6. Predict on New Samples
# -----------------------------
samples = np.array([
    [20, 70, 1.020, 1],
    [20, 70, 1.020, 2],
    [20, 70, 1.020, 3],
    [20, 70, 1.020, 4],
    [20, 70, 1.020, 5],
    [20, 70, 1.020, 6],
], dtype='float32')

# Apply same scaling to samples
samples_scaled = (samples - scaler.mean_) / scaler.scale_
sample_tensor = torch.tensor(samples_scaled, dtype=torch.float32)

with torch.no_grad():
    preds = model(sample_tensor)
    print("\nPredictions:", (preds >= 0.5).int().squeeze().tolist())

print(model.linear.weight)  # Shows the learned weights for each feature
