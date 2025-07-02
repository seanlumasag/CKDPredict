from django.urls import path
from .views import predict_ckd

urlpatterns = [
    path('predict/', predict_ckd),
]
