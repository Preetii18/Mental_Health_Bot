import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load BERT model and tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Streamlit App
st.title("Mental Health Bot - Sentiment Analyzer")

st.write("Enter your message below:")

user_input = st.text_area("Your Message")

if st.button("Analyze Sentiment"):
    if user_input:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs).item() + 1  # 1 to 5 star sentiment

        sentiment = {
            1: "Very Negative",
            2: "Negative",
            3: "Neutral",
            4: "Positive",
            5: "Very Positive"
        }[prediction]

        st.subheader("Result:")
        st.success(f"Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter some text before analyzing.")


