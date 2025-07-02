from django.db import models

class Patient(models.Model):
    age = models.FloatField()
    bp = models.FloatField()
    sg = models.FloatField()
    al = models.FloatField()
    # Add more fields as needed
    prediction = models.BooleanField(null=True)  # CKD True/False
