from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import spacy

# Define input schema
class SMSInput(BaseModel):
    message: str

app = FastAPI()

# Global objects (lazy-loaded)
tokenizer = None
embedding_model = None
svm_classifier = None
nlp = None

@app.on_event("startup")
def load_dependencies():
    global tokenizer, embedding_model, svm_classifier, nlp

    print("🔧 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    embedding_model = AutoModel.from_pretrained("xlm-roberta-base")
    embedding_model.eval()  # put in eval mode to reduce memory usage

    print("🧠 Loading SVM model...")
    svm_classifier = joblib.load("svm_model.pkl")  # pre-trained classifier

    print("🗣️ Loading calamanCy NLP...")
    nlp = spacy.load("tl_calamancy_md")  # Pre-installed via .whl

    print("✅ All models loaded successfully.")

# Feature extraction: Embedding + spaCy
def extract_features(text):
    with torch.no_grad():
        # Tokenize and encode
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = embedding_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

@app.post("/predict")
def predict_sms(input_data: SMSInput):
    try:
        doc = nlp(input_data.message)
        features = extract_features(doc.text).reshape(1, -1)
        prediction = svm_classifier.predict(features)[0]
        return {"prediction": "spam" if prediction == 1 else "ham"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
