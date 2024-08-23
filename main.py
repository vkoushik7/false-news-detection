import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('wordnet')
nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

with open('trained/logistic_regression.pkl', 'rb') as model_file:
    model_lr = pickle.load(model_file)

with open('trained/random_forest.pkl', 'rb') as model_rf_file:
    model_rf = pickle.load(model_rf_file)

with open('trained/tfidf_vectorizer_rf.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

class Preprocessing:
    def __init__(self, data):
        self.data = data
        
    def text_preprocessing_user(self):
        lm = WordNetLemmatizer()
        pred_data = [self.data]    
        preprocess_data = []
        for data in pred_data:
            review = re.sub('[^a-zA-Z0-9]', ' ', data)
            review = review.lower()
            review = review.split()
            review = [lm.lemmatize(x) for x in review if x not in stopwords_set]
            review = " ".join(review)
            preprocess_data.append(review)
        return preprocess_data

def predict_news_log(news_text):
    preprocess_data = Preprocessing(news_text).text_preprocessing_user()
    data = tfidf_vectorizer.transform(preprocess_data)
    prediction = model_lr.predict(data)
    return "The News Is Fake" if prediction[0] == 0 else "The News Is Real"

def predict_news_rf(news_text):
    preprocess_data = Preprocessing(news_text).text_preprocessing_user()
    data = tfidf_vectorizer.transform(preprocess_data)
    prediction = model_rf.predict(data)
    return "The News Is Fake" if prediction[0] == 0 else "The News Is Real"

def main():
    st.title('False News Detection App')

    model_choice = st.radio("Select Model:", ('Random Forest', 'Logistic Regression'))

    news_text = st.text_area('Enter the news text:', '')

    if st.button('Predict'):
        if not news_text.strip():
            st.error("Please enter some text to predict.")
        else:
            if model_choice == 'Random Forest':
                result = predict_news_rf(news_text)
            else:
                result = predict_news_log(news_text)
            st.success(result)

if __name__ == '__main__':
    main()