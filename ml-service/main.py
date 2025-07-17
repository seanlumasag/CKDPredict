from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

class PatientData(BaseModel):
    age: float
    bp: float
    sg: float
    al: float

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

app = FastAPI()

scaler = joblib.load("scaler.pkl")

model = CKDModel()
model.load_state_dict(torch.load("ckd_model.pt"))
model.eval()

@app.post("/predict")
def predict(patient: PatientData):
    features = np.array([[patient.age, patient.bp, patient.sg, patient.al]], dtype=np.float32)

    scaled_features = scaler.transform(features)

    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        prob_ckd = output.item()
        binary_prediction = int(prob_ckd > 0.5)

    return {
        "probability_ckd": prob_ckd,
        "prediction": binary_prediction
    }
    