from django.db import models

class ExtractedText(models.Model):
    text = models.TextField()

    def __str__(self):
        return self.text

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')

    def __str__(self):
        return self.file.name


class BiLSTMClassifier:
    pass