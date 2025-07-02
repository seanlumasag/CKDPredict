from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import PatientSerializer
from .models import Patient
import joblib
import numpy as np
import os

from .dummy_model import model

@api_view(['POST'])
def predict_ckd(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data
        input_features = [
            data['age'],
            data['bp'],
            data['sg'],
            data['al'],
            # Add all other features used in training in correct order
        ]
        prediction = model.predict([input_features])[0]
        data['prediction'] = prediction
        # Optionally save the patient data
        Patient.objects.create(**data)
        return Response({'ckd_risk': bool(prediction)})
    return Response(serializer.errors, status=400)
