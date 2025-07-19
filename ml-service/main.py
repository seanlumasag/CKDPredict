# Import necessary libraries
from fastapi import FastAPI                # FastAPI for creating the web API
from pydantic import BaseModel             # For defining and validating request data models
import torch                               # PyTorch for model inference
import torch.nn as nn                      # For building neural network layers
import joblib                              # To load pre-saved preprocessing objects like scalers
import numpy as np                         # For numerical computations (especially array operations)

# Define the expected format of input data using Pydantic
class PatientData(BaseModel):
    age: float     # Age of patient
    bp: float      # Blood Pressure
    sg: float      # Specific Gravity
    al: float      # Albumin

# Define the architecture of the neural network model (same as used during training)
class CKDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),   # Input layer: 4 features -> 16 neurons
            nn.ReLU(),          # Activation function
            nn.Linear(16, 8),   # Hidden layer: 16 -> 8 neurons
            nn.ReLU(),          # Activation function
            nn.Linear(8, 1),    # Output layer: 8 -> 1 neuron
            nn.Sigmoid()        # Sigmoid to output probability between 0 and 1
        )

    def forward(self, x):
        return self.net(x)      # Defines how input flows through the network

# Initialize FastAPI app
app = FastAPI()

# Load the pre-fitted data scaler used during training
scaler = joblib.load("scaler.pkl")

# Load the trained PyTorch model
model = CKDModel()                                      # Instantiate model with same architecture
model.load_state_dict(torch.load("ckd_model.pt"))       # Load trained weights
model.eval()                                            # Set model to evaluation mode (no gradients)

# Define the /predict POST endpoint for CKD prediction
@app.post("/predict")
def predict(patient: PatientData):
    # Extract patient input and convert to 2D NumPy array
    features = np.array([[patient.age, patient.bp, patient.sg, patient.al]], dtype=np.float32)

    # Scale the features using the same scaler used during training
    scaled_features = scaler.transform(features)

    # Convert scaled features to PyTorch tensor for model input
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    # Perform prediction with the model, disabling gradient calculations
    with torch.no_grad():
        output = model(input_tensor)           # Forward pass through model
        prob_ckd = output.item()               # Get raw probability from output tensor
        binary_prediction = int(prob_ckd > 0.5)  # Convert probability to binary (0 = low risk, 1 = high risk)

    # Return the prediction and probability as JSON response
    return {
        "probability_ckd": prob_ckd,
        "prediction": binary_prediction
    }
