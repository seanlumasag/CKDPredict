from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Patient
from .serializers import PatientSerializer
from .dummy_model import model

@api_view(['POST'])
def predict_ckd(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data
        input_features = [data['age'], data['bp'], data['sg'], data['al']]

        prediction_list = model.predict(input_features)
        prediction = prediction_list[0]

        Patient.objects.create(
            age=data['age'],
            bp=data['bp'],
            sg=data['sg'],
            al=data['al'],
            prediction=bool(prediction)
        )
        return Response({'ckd_risk': bool(prediction)})
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

        input_features = [[data['age'], data['bp'], data['sg'], data['al']]]
        prediction_list = model.predict(input_features)
        prediction = prediction_list[0]

        updated_patient.prediction = bool(prediction)
        updated_patient.save()

        return Response({'ckd_risk': bool(prediction)})
    return Response(serializer.errors, status=400)


@api_view(['DELETE'])
def delete_patient(request, pk):
    try:
        patient = Patient.objects.get(pk=pk)
        patient.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    except Patient.DoesNotExist:
        return Response({'error': 'Patient not found'}, status=status.HTTP_404_NOT_FOUND)