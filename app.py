import streamlit as st
import joblib
import pyttsx3

# Load the model from the .pkl file
model = joblib.load('best_fake_news_model.pkl')

# Initialize pyttsx3 for voice output
engine = pyttsx3.init()

# Streamlit app title
st.title("Fake News Prediction")

# Description
st.write("Enter the text input in the box below:")

# Input text box for the feature
user_input = st.text_input("Enter text input:")

# Function to convert text to speech
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Button to trigger prediction
if st.button("Predict"):
    try:
        # Make prediction (ensure input is in the format your model expects)
        prediction = model.predict([user_input])[0]  # Wrap in a list for single-sample prediction

        # Determine the prediction result as a meaningful text
        result_text = "This is likely fake news." if prediction == 1 else "This is likely real news."

        # Display the result
        st.write("Prediction:", result_text)

        # Trigger voice output
        speak_text(result_text)
        
    except Exception as e:
        error_message = f"Error: {e}"
        st.write(error_message)
        speak_text(error_message)
