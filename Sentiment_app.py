import streamlit as st
import joblib

# Load the saved models
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define a function to predict sentiment
def predict_sentiment(user_text):
    # Transform the user input text using the loaded vectorizer
    vectorized_text = vectorizer.transform([user_text])

    # Predict the sentiment using the loaded model
    predicted_sentiment = model.predict(vectorized_text)

    # Get the actual label from the loaded label encoder
    predicted_label = label_encoder.inverse_transform(predicted_sentiment)[0]

    return predicted_label

# Streamlit UI
st.title("Tweet Sentiment Analysis")

# User input text
user_text = st.text_area("Enter a tweet to analyze sentiment:")

if st.button("Predict Sentiment"):
    if user_text:
        predicted_sentiment = predict_sentiment(user_text)
        st.write(f"Predicted Sentiment: **{predicted_sentiment}**")
    else:
        st.write("Please enter a tweet.")
