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
def get_patients(request):
    patients = Patient.objects.all()
    serializer = PatientSerializer(patients, many=True)
    return Response(serializer.data)
