from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', lambda request: HttpResponse("CKD Predictor API is live!")),
    path('api/', include('predictor.urls')),
]
