from django.forms import forms

from .models import UploadedFile


class UploadFileForm(forms.Form):
    class Meta:
        model = UploadedFile
        fields = ['file']