from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- 1. Initialize Flask App and Load the Model ---
app = Flask(__name__)

# Define an absolute path to the model folder.
# This assumes your app.py is in a 'src' folder and 'models' is in the root.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'saved_models', 'minilm_fake_news_model')


MODEL = None
TOKENIZER = None

try:
    print(f"Attempting to load model from absolute path: {MODEL_PATH}")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
    MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("✅ Model and tokenizer loaded successfully.")
except OSError:
    print(f"❌ Error: Model not found at '{MODEL_PATH}'.")
    print("Please ensure the 'saved_models' folder exists inside a 'models' folder in your project root.")
    print("Also ensure you have run the training script first to create the model.")

# --- 2. The Prediction Function ---
def predict_news(text):
    """
    Predicts if a news text is REAL or FAKE using the loaded model.
    """
    if MODEL is None or TOKENIZER is None:
        return "Model not loaded", 0.0

    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = MODEL(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    predicted_class_id = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_id].item()
    label = "REAL" if predicted_class_id == 0 else "FAKE"
    return label, confidence

# --- 3. Define the Web Page Route ---
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    confidence_score = None
    news_article_text = ""
    error_message = None

    if MODEL is None or TOKENIZER is None:
        error_message = "Model or Tokenizer failed to load. Please check the server logs."

    if request.method == 'POST':
        news_article_text = request.form.get('news_article', '')
        if not news_article_text.strip():
            error_message = "Please enter some news text to analyze."
        elif not error_message:
            label, confidence = predict_news(news_article_text)
            prediction_result = f"{label} NEWS"
            confidence_score = confidence

    return render_template(
        'index.html',
        prediction=prediction_result,
        confidence=confidence_score,
        news_article=news_article_text,
        error=error_message
    )

# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
