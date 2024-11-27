import streamlit as st
import joblib

# Function to load the model and vectorizer
def load_model():
    model = joblib.load('best_fake_news_model.pkl')  # Adjust path if needed
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Adjust path if needed
    return model, tfidf_vectorizer

# Load the model and vectorizer once when the app starts
model, tfidf_vectorizer = load_model()

# Streamlit app layout
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# Add some custom CSS styling to make it colorful
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #FF6347;
    }
    .subheader {
        font-size: 24px;
        color: #4CAF50;
    }
    .prediction {
        font-size: 20px;
        font-weight: bold;
        color: #1E90FF;
    }
    .button {
        background-color: #FF6347;
        color: white;
        font-weight: bold;
    }
    .text-box {
        border: 2px solid #1E90FF;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a title to the app
st.markdown('<p class="title">Fake News Detection</p>', unsafe_allow_html=True)
st.write("Enter news text below to check if it's Real or Fake.")

# Input text box with improved styling
news_text = st.text_area("News Text", "", height=200, key="news_text", help="Paste the news text here", placeholder="Enter the news text...")

# Add a prediction button with custom style
if st.button("Check News", key="predict_button"):
    if news_text.strip() != "":
        # Transform and predict
        text_vector = tfidf_vectorizer.transform([news_text])
        prediction = model.predict(text_vector)
        result = "Real News" if prediction[0] == 1 else "Fake News"
        
        # Display the result in a colorful and bold style
        st.markdown(f'<p class="prediction">Prediction: {result}</p>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze.", icon="⚠️")

# Footer
st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

