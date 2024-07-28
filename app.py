import streamlit as st
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

nltk.download('punkt')
nltk.download('stopwords')

# Preprocess function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

# Function to predict category
def predict_category(query):
    query = preprocess_text(query)
    query_vec = vectorizer.transform([query])
    start_time = time.time()
    category = model.predict(query_vec)
    end_time = time.time()
    response_time = end_time - start_time
    return category[0], response_time

# Streamlit UI
st.title("Query Category Prediction")

user_query = st.text_input("Enter your query:")

if st.button("Predict"):
    if user_query:
        category, response_time = predict_category(user_query)
        st.write(f'The predicted category for the query "{user_query}" is "{category}".')
        st.write(f"Response time: {response_time:.4f} seconds")
        st.write("Status: Success")
    else:
        st.write("Please enter a query.")
