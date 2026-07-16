import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_sentiment_model():
    return tf.keras.models.load_model("models/imdb_rnn.keras")

model = load_sentiment_model()

word_index = imdb.get_word_index()

MAX_LEN = 500

VOCAB_SIZE = 10000  # must match the vocab size used during training

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    encoded = [word_index.get(word, 2) + 3 for word in words]

    padded = pad_sequences([encoded], maxlen=MAX_LEN)

    return padded

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the model will predict whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Movie Review",
    height=180,
    placeholder="Type your review here..."
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        processed = preprocess_text(review)

        prediction = model.predict(processed, verbose=0)[0][0]

        if prediction >= 0.5:

            st.success("😊 Positive Review")
            st.metric("Confidence", f"{prediction:.2%}")

        else:

            st.error("😞 Negative Review")
            st.metric("Confidence", f"{1-prediction:.2%}")