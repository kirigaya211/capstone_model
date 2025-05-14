from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import calamancy
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, Any

app = FastAPI()


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

def generate_visualization(features, prediction, message, processed_text):
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
 
    feature_names = ['Numeric Sender', 'Short Sender'] + [f'Embedding {i+1}' for i in range(len(features)-2)]
    feature_importance = np.abs(features)
    sorted_idx = np.argsort(feature_importance)
    
    ax1.barh(range(len(feature_importance)), feature_importance[sorted_idx])
    ax1.set_yticks(range(len(feature_importance)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax1.set_title('Feature Importance Analysis')
    ax1.set_xlabel('Absolute Feature Value')
    
    # Plot 2: Text analysis
    words = processed_text.split()
    word_lengths = [len(word) for word in words]
    ax2.bar(range(len(words)), word_lengths)
    ax2.set_title('Word Length Analysis')
    ax2.set_xlabel('Word Position')
    ax2.set_ylabel('Word Length')
    

    plt.figtext(0.5, 0.01, 
                f'Prediction: {prediction}\n'
                f'Original Message: {message}\n'
                f'Processed Text: {processed_text}',
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    

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
    visualization: str

@app.post("/predict")
def predict(data: SMSRequest):
    processed = preprocess_text(data.message)
    embedding = get_embeddings([processed])[0]
    sender_features = get_sender_features(data.sender)
    features = np.concatenate([embedding, sender_features])
    # Optionally: features = (features - feature_means)  # If you want to mean-center
    prediction = svm_model.predict([features])[0]
    label = "SPAM" if prediction == 1 else "HAM"
    return {
        "sender": data.sender,
        "message": data.message,
        "processed": processed,
        "prediction": label
    }

@app.post("/predict_with_visualization", response_model=PredictionResponse)
def predict_with_visualization(data: SMSRequest):
    processed = preprocess_text(data.message)
    embedding = get_embeddings([processed])[0]
    sender_features = get_sender_features(data.sender)
    features = np.concatenate([embedding, sender_features])
    prediction = svm_model.predict([features])[0]
    label = "SPAM" if prediction == 1 else "HAM"
    

    visualization = generate_visualization(features, label, data.message, processed)
    
    return {
        "sender": data.sender,
        "message": data.message,
        "processed": processed,
        "prediction": label,
        "visualization": visualization
    }
