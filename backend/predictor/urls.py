from django.urls import path
from .views import predict_ckd, recent_predictions, update_patient

urlpatterns = [
    path('predict/', predict_ckd),
    path('recent/', recent_predictions),
    path('patient/<int:pk>/', update_patient),

]
