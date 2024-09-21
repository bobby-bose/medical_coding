import spacy
from django.views import View
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.http import HttpResponse

from .models import ExtractedText
import PyPDF2


class UploadPDFView(View):
    def extract_text_from_pdf(self, pdf_file):
        text = ""
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
        return text

    def post(self, request):
        if 'file' in request.FILES:
            uploaded_file = request.FILES['file']
            try:
                # Extract text from the uploaded PDF file
                extracted_text = self.extract_text_from_pdf(uploaded_file)

                # Store extracted text in the database
                extracted_obj = ExtractedText.objects.create(text=extracted_text)
                extracted_obj.save()

                return redirect('result')  # Redirect to result page
            except Exception as e:
                messages.error(request, f"Error: {str(e)}")
                return redirect('upload_pdf')  # Redirect back to upload page in case of error
        else:
            messages.error(request, 'No file uploaded')
            return redirect('upload_pdf')  # Redirect back to upload page if no file is uploaded

    def get(self, request):
        return render(request, 'upload.html')

class ResultView(View):
    def get(self, request):
        extracted_obj = ExtractedText.objects.last()  # Get the last inserted object
        if extracted_obj:
            extracted_text = extracted_obj.text
            list_extracted_text = extracted_text.split('\n')
            return render(request, 'result.html', {'text': str(list_extracted_text)})
        else:
            return HttpResponse("No extracted text found")




class CompareDescriptionView(View):
    def get(self, request):
        # Retrieve the last inserted text from the database
        extracted_obj = ExtractedText.objects.last()
        print(extracted_obj)
        if extracted_obj:
            extracted_text = extracted_obj.text

            # Split the extracted text into sentences
            sentences = extracted_text.split('.')

            # Load the spacy model
            nlp = spacy.load("en_core_web_sm")

            # List of medical terms to compare with
            medical_terms = medical_terms

            # Initialize a list to store matching sentences
            matching_sentences = []

            # Iterate over each sentence
            for sentence in sentences:
                doc = nlp(sentence)
                # Check if any medical term exists in the sentence
                for token in doc:
                    if token.text.lower() in medical_terms:
                        matching_sentences.append(sentence.strip())
                        break

            # Pass the matching sentences to the template
            context = {'matching_sentences': matching_sentences}
            print(matching_sentences)
            return render(request, 'compare.html', context)
        else:
            return render(request, 'compare.html')


class HomeView(View):
    def get(self, request):
        return render(request, 'home.html')

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
            return redirect('home')
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
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('login')

    def get(self, request):
        return render(request, 'login.html')

class LogoutUserView(View):
    def get(self, request):
        logout(request)
        return redirect('login')



#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
import warnings
import matplotlib
import matplotlib.pyplot as plt
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

random_state_number = 8888
df = pd.read_csv('mtsamples.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)


def trim(df):
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df


df = trim(df)
df = df[df['medical_specialty'].isin(['Neurosurgery', 'ENT - Otolaryngology', 'Discharge Summary'])]


def remove_punc_num(df, attribute):
    df.loc[:, attribute] = df[attribute].apply(lambda x: " ".join(re.findall('[\w]+', x)))
    df[attribute] = df[attribute].str.replace('\d+', '')
    return df


df = remove_punc_num(df, 'transcription')


def tokenise(df, attribute):
    tk = WhitespaceTokenizer()
    df['tokenised'] = df.apply(lambda row: tk.tokenize(str(row[attribute])), axis=1)
    return df


df = tokenise(df, 'transcription')

from nltk.stem.snowball import SnowballStemmer


def stemming(df, attribute):
    stemmer = SnowballStemmer("english")
    df['stemmed'] = df[attribute].apply(lambda x: [stemmer.stem(y) for y in x])
    return df


df = stemming(df, 'tokenised')


def remove_stop_words(df, attribute):
    stop = stopwords.words('english')
    df['stemmed_without_stop'] = df[attribute].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))
    return df


df = remove_stop_words(df, 'stemmed')

total_word_count_normalised = df['stemmed_without_stop'].str.split().str.len().sum()
le = preprocessing.LabelEncoder()
le.fit(df['medical_specialty'])
df['encoded_target'] = le.transform(df['medical_specialty'])

n_gram_features = {'unigram': (1, 1), 'unigram_bigram': (1, 2), 'bigram': (2, 2),
                   'bigram_trigram': (2, 3), 'trigram': (3, 3)}


def generate_n_gram_features(flat_list_transcription):
    temp = []
    for key, values in n_gram_features.items():
        vectorizer = CountVectorizer(ngram_range=values)
        vectorizer.fit(flat_list_transcription)
        temp.append(vectorizer.transform(flat_list_transcription))
    return temp


flat_list_transcription = df['stemmed_without_stop'].values.tolist()
temp = generate_n_gram_features(flat_list_transcription)

dataframes = {'unigram': temp[0], 'unigram_bigram': temp[1], 'bigram': temp[2],
              'bigram_trigram': temp[3], 'trigram': temp[4]}


def get_performance(param_grid, base_estimator, dataframes):
    df_name_list = [];
    best_estimator_list = [];
    best_score_list = [];
    test_predict_result_list = [];
    metric_list = [];

    for df_name, df in dataframes.items():

        X_train, X_test, y_train, y_test = train_test_split(df, df_target, test_size=0.2,
                                                            random_state=random_state_number)
        for _, metric_dict in metrics.items():
            sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5, scoring=metric_dict[1],
                                     random_state=random_state_number,
                                     factor=2).fit(X_train, y_train)

            best_estimator = sh.best_estimator_
            clf = best_estimator.fit(X_train, y_train)
            prediction = clf.predict(X_test)
            test_predict_result = metric_dict[0](y_test, prediction, average='macro')

            df_name_list.append(df_name);
            best_estimator_list.append(best_estimator);
            best_score_list.append(sh.best_score_);
            test_predict_result_list.append(test_predict_result);
            metric_list.append(metric_dict[1])

    model_result = pd.DataFrame({'Vector': df_name_list, 'Metric': metric_list,
                                 'Calibrated Estimator': best_estimator_list,
                                 'Best CV Metric Score': best_score_list,
                                 'Test Predict Metric Score': test_predict_result_list})
    return model_result


param_grid = {'max_depth': [None, 30, 32, 35, 37, 38, 39, 40], 'min_samples_split': [2, 150, 170, 180, 190, 200]}
base_estimator = RandomForestClassifier(random_state=random_state_number, n_jobs=-1)

model_result = get_performance(param_grid, base_estimator, dataframes)

best_estimator_chosen = model_result.sort_values('Test Predict Metric Score', ascending=False).groupby(
    'Vector').first().reset_index()
best_estimator_chosen

df_target = df['encoded_target']
metrics = {'Accuracy': (accuracy_score, 'accuracy'),
           'Precision': (precision_score, 'precision_macro'),
           'Recall': (recall_score, 'recall_macro'),
           'F1 Score': (f1_score, 'f1_macro')}

X_train, X_test, y_train, y_test = train_test_split(dataframes['unigram_bigram'], df_target, test_size=0.2,
                                                    random_state=random_state_number)

clf = RandomForestClassifier(max_depth=30, min_samples_split=180, random_state=random_state_number, n_jobs=-1).fit(
    X_train, y_train)
prediction = clf.predict(X_test)

print(classification_report(y_test, prediction))
