from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
import calamancy
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any

app = FastAPI()

# --- Load NLP tools and model ---
try:
    nlp = calamancy.load("tl_calamancy_md-0.1.0")
except Exception as e:
    print(f"Error loading calamancy model: {e}")
    print("Attempting to download model...")
    os.system("pip install calamancy")
    nlp = calamancy.load("tl_calamancy_md-0.1.0")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
transformer_model = AutoModel.from_pretrained("xlm-roberta-base")
svm_model = joblib.load("svm_model.pkl")
# feature_means = np.load("svm_feature_means.npy")  # Optional: for scaling

def preprocess_text(text):
    text = str(text)
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_punct and not token.is_space and token.lemma_.isalpha()
    ]
    return " ".join(tokens) if tokens else text.lower()

def get_embeddings(text_list):
    transformer_model.eval()
    with torch.no_grad():
        encoded = tokenizer(text_list, return_tensors='pt', truncation=True, padding=True, max_length=128)
        output = transformer_model(**encoded)
        cls_embeddings = output.last_hidden_state[:, 0, :]
        return cls_embeddings.numpy()

def get_sender_features(sender):
    is_numeric = int(str(sender).isdigit())
    is_short = int(len(str(sender)) < 6)
    return np.array([is_numeric, is_short])

def generate_explanation_plot(features, prediction):
    # Create a SHAP explainer
    explainer = shap.KernelExplainer(svm_model.predict_proba, np.zeros((1, features.shape[0])))
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(features.reshape(1, -1))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[1], features.reshape(1, -1), plot_type="bar", show=False)
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str

class SMSRequest(BaseModel):
    sender: str
    message: str

class PredictionResponse(BaseModel):
    sender: str
    message: str
    processed: str
    prediction: str
    explanation_plot: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict(data: SMSRequest):
    processed = preprocess_text(data.message)
    embedding = get_embeddings([processed])[0]
    sender_features = get_sender_features(data.sender)
    features = np.concatenate([embedding, sender_features])
    
    # Get prediction and probability
    prediction = svm_model.predict([features])[0]
    probabilities = svm_model.predict_proba([features])[0]
    confidence = float(probabilities[1] if prediction == 1 else probabilities[0])
    
    # Generate explanation plot
    explanation_plot = generate_explanation_plot(features, prediction)
    
    return {
        "sender": data.sender,
        "message": data.message,
        "processed": processed,
        "prediction": "SPAM" if prediction == 1 else "HAM",
        "explanation_plot": explanation_plot,
        "confidence": confidence
    }
