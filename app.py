import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ! ---------------------- NLTK SETUP ----------------------
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ! ---------------------- TEXT PREPROCESSING ----------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))     # optimized

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ! ---------------------- SAFE MODEL LOADING ----------------------
if not os.path.exists('vectorizer.pkl') or not os.path.exists('model.pkl'):
    st.error("Model files not found. Please train and save them again.")
    st.stop()

if os.path.getsize('vectorizer.pkl') < 100 or os.path.getsize('model.pkl') < 100:
    st.error("Model files are corrupted. Please retrain the model.")
    st.stop()

try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ! ---------------------- UI ----------------------
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # ! 1. preprocess
        transformed_sms = transform_text(input_sms)

        # ! 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # ! 3. predict
        result = model.predict(vector_input)[0]

        # ! 4. display result
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")