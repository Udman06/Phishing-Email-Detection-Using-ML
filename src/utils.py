# src/utils.py

import pickle
import os

def load_model_and_vectorizer():
    with open("model/phishing_detector.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("model/vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def predict_emaildet(email_text):
    model, vectorizer = load_model_and_vectorizer()
    features = vectorizer.transform([email_text])
    prediction = model.predict(features)[0]
    return "Phishing" if prediction == 1 else "Legitimate"