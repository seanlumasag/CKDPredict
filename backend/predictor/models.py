from django.db import models

class Patient(models.Model):
    age = models.FloatField()
    bp = models.FloatField()
    sg = models.FloatField()
    al = models.FloatField()
    prediction = models.BooleanField(null=True) 
    created_at = models.DateTimeField(auto_now_add=True) 
