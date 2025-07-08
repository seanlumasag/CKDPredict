import torch
import torch.nn as nn
import joblib
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Patient
from .serializers import PatientSerializer

# Define model architecture (must match training)
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

# Instantiate model and load weights
model = CKDModel()
model.load_state_dict(torch.load('predictor/ckd_model.pt', map_location='cpu'))
model.eval()

# Load scaler saved from training
scaler = joblib.load('predictor/scaler.pkl')

@api_view(['POST'])
def predict_ckd(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data

        # Extract features, scale and convert to tensor
        input_data = [[data['age'], data['bp'], data['sg'], data['al']]]
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        # Model prediction
        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()
            prediction = int(probability >= 0.5)

        # Save patient with prediction
        Patient.objects.create(
            age=data['age'],
            bp=data['bp'],
            sg=data['sg'],
            al=data['al'],
            prediction=bool(prediction)
        )

        return Response({'ckd_risk': bool(prediction), 'confidence': round(probability, 3)})

    return Response(serializer.errors, status=400)


@api_view(['GET'])
def recent_predictions(request):
    patients = Patient.objects.order_by('-created_at')[:10]
    serializer = PatientSerializer(patients, many=True)
    return Response(serializer.data)


@api_view(['PUT'])
def update_patient(request, pk):
    try:
        patient = Patient.objects.get(pk=pk)
    except Patient.DoesNotExist:
        return Response({'error': 'Not found'}, status=404)

    serializer = PatientSerializer(patient, data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data
        updated_patient = serializer.save()

        # Scale input and predict
        input_data = [[data['age'], data['bp'], data['sg'], data['al']]]
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)

        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()
            prediction = int(probability >= 0.5)

        updated_patient.prediction = bool(prediction)
        updated_patient.save()

        return Response({'ckd_risk': bool(prediction), 'confidence': round(probability, 3)})

    return Response(serializer.errors, status=400)


@api_view(['DELETE'])
def delete_patient(request, pk):
    try:
        patient = Patient.objects.get(pk=pk)
        patient.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found'}, status=status.HTTP_404_NOT_FOUND)
