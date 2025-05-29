from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import calamancy
from transformers import AutoTokenizer, AutoModel
import numpy as np

app = FastAPI()

# --- Load NLP tools and model ---
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

class SMSRequest(BaseModel):
    sender: str
    message: str

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
