import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load max_len
with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Same preprocessing (VERY IMPORTANT)
def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Prediction function
def generate_text(seed_text, next_words):
    for _ in range(next_words):
        seq = tokenizer.texts_to_sequences([clean_text(seed_text)])[0]
        seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')

        predicted = np.argmax(model.predict(seq, verbose=0), axis=1)[0]

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break

    return seed_text

# UI
st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput input {
        border-radius: 10px;
        padding: 10px;
    }
    .stButton button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("✨ Next Word Predictor")
st.caption("Generate intelligent sentence completions using LSTM")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("Enter text:")
    user_input = st.text_input("", label_visibility="collapsed")

with col2:
    st.markdown("Words:")
    num_words = st.number_input("", min_value=1, max_value=20, value=5, label_visibility="collapsed")

st.divider()

if st.button("Predict"):
    if user_input:
        with st.spinner("Generating..."):
            result = generate_text(user_input, num_words)
            st.markdown(
                f"""
                <div style="background-color:#1e3a2f;padding:15px;border-radius:10px;">
                    <h4 style="color:#00ff9c;">{result}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("Please enter some text")