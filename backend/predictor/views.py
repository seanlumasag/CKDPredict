from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Patient
from .serializers import PatientSerializer


@api_view(['POST'])
def predict_ckd(request):
    serializer = PatientSerializer(data=request.data)
    if serializer.is_valid():
        data = serializer.validated_data
        input_features = [data['age'], data['bp'], data['sg'], data['al']]

        # Dummy prediction logic
        prediction = 1 if data['age'] > 50 else 0

        # Save to DB
        patient = Patient.objects.create(
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
    patients = Patient.objects.order_by('-created_at')[:10]  # last 10 entries
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
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=400)
