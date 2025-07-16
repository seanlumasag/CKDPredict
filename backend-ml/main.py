from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

# Define request schema
class PatientData(BaseModel):
    age: float
    bp: float
    sg: float
    al: float

# Define the same model architecture
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

# Initialize FastAPI app
app = FastAPI()

# Load scaler and model once on startup
scaler = joblib.load("scaler.pkl")

model = CKDModel()
model.load_state_dict(torch.load("ckd_model.pt"))
model.eval()

@app.post("/predict")
def predict(patient: PatientData):
    # Convert input data to numpy array
    features = np.array([[patient.age, patient.bp, patient.sg, patient.al]], dtype=np.float32)

    # Scale features using saved scaler
    scaled_features = scaler.transform(features)

    # Convert to torch tensor
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    # Predict with model (no grad for inference)
    with torch.no_grad():
        output = model(input_tensor)
        prob_ckd = output.item()               # float between 0 and 1
        binary_prediction = int(prob_ckd > 0.5)  # convert to 0 or 1

    return {
        "probability_ckd": prob_ckd,
        "prediction": binary_prediction
    }
    