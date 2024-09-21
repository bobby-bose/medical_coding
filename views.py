import re
from collections import Counter

import spacy
from django.views import View
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import HttpResponse
from nltk.stem.snowball import SnowballStemmer
from .forms import UploadFileForm
from .models import ExtractedText
import PyPDF2
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from django.shortcuts import render
from django.views import View
from .models import ExtractedText
from .stop import aysha_stop_words

class RegisterUserView(View):
    def post(self, request):
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password1']
        password2 = request.POST['password2']

        if password == password2:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username is already taken')
                return redirect('register')
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            login(request, user)
            return redirect('upload_pdf')
        else:
            messages.error(request, 'Passwords do not match')
            return redirect('register')

    def get(self, request):
        return render(request, 'register.html')

class LoginUserView(View):
    def post(self, request):
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('upload_pdf')
        else:
            messages.error(request, 'Invalid username or password')
            print("Invalid username or password")
            return redirect('login')

    def get(self, request):
        return render(request, 'login.html')

class LogoutUserView(View):
    def get(self, request):
        logout(request)
        return redirect('login')

class UploadPDFView(View):
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
        return text

    def get(self, request):
        form = UploadFileForm()
        return render(request, 'home.html', {'form': form})

    def post(self, request):
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_file = request.FILES['file']
            try:
                extracted_text = self.extract_text_from_pdf(pdf_file)
                extracted_obj = ExtractedText.objects.create(text=extracted_text)
                extracted_obj.save()
                return redirect('result')
            except Exception as e:
                messages.error(request, f"Error: {str(e)}")
        else:
            messages.error(request, 'Invalid form data')
        return redirect('result')

class ResultView(View):
    def get(self, request):
        extracted_obj = ExtractedText.objects.last()  # Get the last inserted object
        if extracted_obj:
            extracted_text = extracted_obj.text
            list_extracted_text = extracted_text.split(' ')
            filtered_text = [word for word in list_extracted_text if word.strip().lower() not in aysha_stop_words]
            print("filtered_text",filtered_text)
            return render(request, 'result.html', {'text': ' '.join(filtered_text)})
        else:
            return HttpResponse("No extracted text found")

import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
class TextProcessingView(View):
    def clean_text(self, text):
        cleaned_tokens = []
        # Tokenize the text
        tokens = word_tokenize(text.lower())  # Convert text to lowercase for case-insensitive matching

        # Remove stopwords and punctuation, and keep alphanumeric characters
        stop_words = set(stopwords.words('english'))
        cleaned_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        print("cleaned_tokens",cleaned_tokens)

        return cleaned_tokens

    def get_icd_codes(self, cleaned_text):
        # Load the JSON file containing ICD codes
        with open('G:/2024 projects/aysha/medical coding/myproject/myapp/output.json', 'r') as json_file:
            icd_codes = json.load(json_file)

        # Create a list to store matching details
        matching_details = []

        # Identify exact matches between cleaned tokens and ICD code descriptions
        for token in cleaned_text:
            for code, details in icd_codes.items():
                for description in details:
                    if (token == description['disease_name'].lower() or  # Check if token matches disease_name
                            token in description['description'].lower().split()):  # Check if token is in description
                        matching_details.append({
                            "icd_code": code,
                            "description": description["description"],
                            "disease_name": description["disease_name"]
                        })
                        break  # Break out of inner loop once a match is found
                else:
                    continue  # Continue to the next code if no match is found in the current code's descriptions
                break  # Break out of the outer loop once a match is found

        return matching_details

    def get(self, request):
        extracted_obj = ExtractedText.objects.last()
        if extracted_obj:
            # Clean the extracted text
            cleaned_text = self.clean_text(extracted_obj.text)

            # Get the ICD codes for exact matches
            icd_codes = self.get_icd_codes(cleaned_text)

            return render(request, 'icd_codes.html', {'identified_codes': icd_codes})
        else:
            return HttpResponse("No extracted text found")



from django.shortcuts import render





def extract_features(cleaned_text):
    return len(cleaned_text)




def tokenize_and_encode(cleaned_text):
    nlp = spacy.load('en_core_web_sm')
    tokens = nlp(' '.join(cleaned_text))
    encoded_tokens = [token.vector for token in tokens]
    return encoded_tokens




from django.views import View
import joblib  # For saving and loading sklearn models
import torch  # For PyTorch models

# Import your Random Forest model

from .models import BiLSTMClassifier
bilstm_model = BiLSTMClassifier()

# Import your biLSTM model
# Assuming you have a PyTorch model class named BiLSTMClassifier
class BiLSTMView(View):
    def get(self, request):
        extracted_obj = ExtractedText.objects.last()
        if extracted_obj:
            cleaned_text = self.clean_text(extracted_obj.text)
            input_sequence = tokenize_and_encode(cleaned_text)
            input_tensor = torch.tensor(input_sequence).unsqueeze(0)
            with torch.no_grad():
                output = bilstm_model(input_tensor)
                _, predicted = torch.max(output, 1)
                predictions = predicted.tolist()
            final=[]
            for i in predictions:
                if i==0:
                    final.append("No")
                else:
                    final.append("Yes")
            return render(request, 'bilstm.html', {'predictions': predictions})
        else:
            return HttpResponse("No extracted text found")



class RandomForestView(View):
    def get(self, request, random_forest_model=None):
        extracted_obj = ExtractedText.objects.last()
        if extracted_obj:
            cleaned_text = self.clean_text(extracted_obj.text)
            features = extract_features(cleaned_text)
            predictions = random_forest_model.predict(features)
            return render(request, 'random_forest.html', {'predictions': predictions})
        else:
            return HttpResponse("No extracted text found")






