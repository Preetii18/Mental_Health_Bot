from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

app = Flask(__name__)

# Load BERT model and tokenizer (can take a few seconds the first time)
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['user_input']

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

    return render_template('result.html', text=user_input, result=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
