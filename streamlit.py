import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import contractions
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
# Download the necessary NLTK models
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Load TF-IDF Vectorizer using caching
@st.cache_resource
def load_tfidf_vectorizer():
    return joblib.load('tfidf_vectorizer.pkl')

tfidf_vectorizer = load_tfidf_vectorizer()

# Lazy load models when needed
@st.cache_resource
def load_model(model_name):
    model_file = {
        "SVM": "svm_model.pkl",
        "Random Forest": "rf_model.pkl",
        "Naive Bayes": "nb_model.pkl"
    }.get(model_name, "svm_model.pkl")  # Default to SVM if model_name not found
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

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a movie review, and select the model to predict the sentiment.")

# Text input
user_input = st.text_area("Enter your review here:", "")

# Model selection
model_option = st.selectbox("Select a model:", ("SVM", "Random Forest", "Naive Bayes"))

# Prediction
if st.button("Predict Sentiment"):
    if model_option in ["SVM", "Random Forest", "Naive Bayes"]:
        prediction = predict_sentiment(user_input, model_option)
        st.write(f"The predicted sentiment is: **{prediction}**")
