import os
import re
from urllib.parse import parse_qs, urlparse

import contractions
import gdown
import joblib
import nltk
import numpy as np
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

# Download the necessary NLTK models
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
# # Helper function to download file from Google Drive
def download_from_google_drive(drive_id, destination):
    # Check if drive_id is a full URL or just an ID
    if drive_id.startswith('http'):
        parsed_url = urlparse(drive_id)
        params = parse_qs(parsed_url.query)
        file_id = params.get('id', [None])[0]
        if not file_id:
            raise ValueError("Could not extract file ID from the provided URL")
    else:
        file_id = drive_id

    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        output = gdown.download(url, destination, quiet=False)
        if output is None:
            print(f"Failed to download file with ID: {file_id}")
            return False
        print(f"File successfully downloaded to {output}")
        return True
    except Exception as e:
        print(f"An error occurred while downloading: {str(e)}")
        return False


# Download models and TF-IDF vectorizer
@st.cache_resource
def load_tfidf_vectorizer():
    # Replace with your actual Google Drive ID
    tfidf_drive_id = '1kavw9Dwgtgwuy4gVDdTtkUpcAeACw68I'
    if not os.path.exists('tfidf_vectorizer.pkl'):
        download_from_google_drive(tfidf_drive_id, 'tfidf_vectorizer.pkl')
        
    return joblib.load('tfidf_vectorizer.pkl')

tfidf_vectorizer = load_tfidf_vectorizer()

@st.cache_resource
def load_model(model_name):
    drive_ids = {
        "SVM": "1etPvpOO6qyHEQgpND93mf5wG52zgbGQv",
        "Random Forest": "1AVP7XbqaRjMb5qE3kc829jA_FweLH_aO",
        "Naive Bayes": "1iauulrpF2FISna1PhaGQay83y1c3fZ0X"
    }
    model_file = {
        "SVM": "svm_model.pkl",
        "Random Forest": "rf_model.pkl",
        "Naive Bayes": "nb_model.pkl"
    }.get(model_name, "svm_model.pkl")
    
    # Download the model file if not already downloaded
    if not os.path.exists(model_file):
        download_from_google_drive(drive_ids[model_name], model_file)
    
    return joblib.load(model_file)

# Preprocessing functions
def handle_negations(text):
    negation_words = {"not", "n't", "never", "no", "neither", "nor"}
    tokens = word_tokenize(text)
    negated_tokens = []
    negation_active = False
    
    for i in range(len(tokens)):
        token = tokens[i].lower()

        if token in negation_words:
            negation_active = True
        elif token in {'.', ',', ';', '!', '?'}:
            negation_active = False

        if negation_active and i+1 < len(tokens):
            negated_tokens.append(f"NOT_{tokens[i+1]}")
            negation_active = False
            continue
        else:
            negated_tokens.append(tokens[i])

    return ' '.join(negated_tokens)

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\[.*?\]', '', text)
    text = contractions.fix(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower()
    text = handle_negations(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = ' '.join(tokens)
    return text

# Prediction function
def predict_sentiment(manual_text, model_name):
    model = load_model(model_name)
    preprocessed_text = preprocess_text(manual_text)
    text_features = tfidf_vectorizer.transform([preprocessed_text])
    sentiment = model.predict(text_features)
    sentiment_label = 'positive' if sentiment[0] == 0 else 'negative'
    return sentiment_label

# Ensure that the model option is stored in session state
if "model_option" not in st.session_state:
    st.session_state.model_option = "SVM" 
# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a movie review, and select the model to predict the sentiment.")

# Text input
user_input = st.text_area("Enter your review here:", "")

# Model selection (default to SVM)
model_option = st.selectbox(
    "Select a model:",
    ("SVM", "Random Forest", "Naive Bayes"),
    index=("SVM", "Random Forest", "Naive Bayes").index(st.session_state.model_option)
)

# Update session state when model selection changes
st.session_state.model_option = model_option

# Prediction
if st.button("Predict Sentiment"):
    prediction = predict_sentiment(user_input, st.session_state.model_option)
    st.write(f"The predicted sentiment is: **{prediction}**")
