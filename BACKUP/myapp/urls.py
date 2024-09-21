from django.urls import path
from .views import (
    UploadPDFView,
    ResultView,
    CompareDescriptionView,
    HomeView,
    RegisterUserView,
    LoginUserView,
    LogoutUserView,
)

urlpatterns = [
    path('upload/', UploadPDFView.as_view(), name='upload_pdf'),
    path('result/', ResultView.as_view(), name='result'),
    path('compare/', CompareDescriptionView.as_view(), name='compare_description'),
    path('home/', HomeView.as_view(), name='home'),
    path('register/', RegisterUserView.as_view(), name='register'),
    path('login/', LoginUserView.as_view(), name='login'),
    path('logout/', LogoutUserView.as_view(), name='logout'),
]
