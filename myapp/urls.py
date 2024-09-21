from django.urls import path
from .views import (
    UploadPDFView,
    ResultView,
TextProcessingView,

    RegisterUserView,
    LoginUserView,
    LogoutUserView,
)

urlpatterns = [
    path('upload/', UploadPDFView.as_view(), name='upload_pdf'),
    path('result/', ResultView.as_view(), name='result'),
    path('text_processing/', TextProcessingView.as_view(), name='text_processing'),

    path('register/', RegisterUserView.as_view(), name='register'),
    path('', LoginUserView.as_view(), name='login'),
    path('logout/', LogoutUserView.as_view(), name='logout'),
]
