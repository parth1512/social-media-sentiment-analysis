
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib

# Set matplotlib backend to Agg to allow headless operations
matplotlib.use('Agg')

def clean_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r"@\w+", '', text)  # Remove mentions
    text = re.sub(r"#\w+", '', text)  # Remove hashtags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r"\d+", '', text)  # Remove digits
    return text.strip()

def load_and_train_model():
    # Load Data
    try:
        df = pd.read_csv('bot_detection_data.csv')
    except FileNotFoundError:
        return None, None, None, "Error: bot_detection_data.csv not found."

    df['Tweet'] = df['Tweet'].str.strip()
    df = df[['Tweet', 'Bot Label']].dropna()
    df['Bot Label'] = df['Bot Label'].astype(int)
    
    # Cleaning
    df['clean_text'] = df['Tweet'].apply(clean_text)

    if df.empty:
        return None, None, None, "Error: DataFrame empty after cleaning."

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['Bot Label'], test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, vectorizer, report, conf_matrix

def predict_single(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return "Bot" if prediction == 1 else "Human"
