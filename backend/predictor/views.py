import torch
import torch.nn as nn
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Patient
from .serializers import PatientSerializer

# Re-define model architecture (must match the one in training)
class CKDModel(nn.Module):
    def __init__(self):
        super(CKDModel, self).__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Instantiate and load state_dict
model = CKDModel()
model.load_state_dict(torch.load('predictor/ckd_model.pt', map_location='cpu'))
model.eval()


@api_view(['POST'])
def predict_ckd(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data

        # Prepare input as tensor (batch of 1 sample)
        input_tensor = torch.FloatTensor([[data['age'], data['bp'], data['sg'], data['al']]])

        # Run prediction
        with torch.no_grad():
            output = model(input_tensor)  # raw logits
            probability= output.item()
            prediction = int(probability >= 0.5)  # binary classification

        # Save to database
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

        input_tensor = torch.FloatTensor([[data['age'], data['bp'], data['sg'], data['al']]])
        with torch.no_grad():
            output = model(input_tensor)
            probability= output.item()
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
