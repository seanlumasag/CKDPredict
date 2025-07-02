from django.urls import path
from .views import predict_ckd, get_patients

urlpatterns = [
    path('predict/', predict_ckd),
    path('patients/', get_patients),

]
