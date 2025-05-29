from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
import calamancy
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk
from nltk.corpus import stopwords

app = FastAPI()

# --- Load NLP tools and model ---
nlp = calamancy.load("tl_calamancy_md-0.1.0")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
transformer_model = AutoModel.from_pretrained("xlm-roberta-base")
svm_model = joblib.load("svm_model.pkl")
# feature_means = np.load("svm_feature_means.npy")  # Optional: for scaling

# --- Load stopwords ---
stop_words = stopwords.words('english')
tagalog_stopwords = stop_words + [
    "ako", "sa", "akin", "ko", "aking", "sarili", "kami", "atin", "ang", "aming", "amin", "ating",
    "ka", "iyong", "iyo", "inyong", "siya", "kanya", "mismo", "ito", "nito", "kanyang", "sila",
    "nila", "kanila", "kanilang", "kung", "ano", "alin", "sino", "kanino", "na", "mga", "iyon",
    "am", "ay", "maging", "naging", "mayroon", "may", "nagkaroon", "pagkakaroon", "gumawa",
    "ginagawa", "ginawa", "paggawa", "ibig", "dapat", "maaari", "marapat", "kong", "ikaw",
    "tayo", "hindi", "namin", "gusto", "nais", "niyang", "nilang", "niya", "huwag", "ginawang",
    "gagawin", "maaaring", "sabihin", "narito", "kapag", "ni", "nasaan", "bakit", "paano",
    "kailangan", "walang", "katiyakan", "isang", "at", "pero", "o", "dahil", "bilang", "hanggang",
    "habang", "ng", "pamamagitan", "para", "tungkol", "laban", "pagitan", "panahon", "bago",
    "pagkatapos", "itaas", "ibaba", "mula", "pataas", "pababa", "palabas", "ibabaw", "ilalim",
    "muli", "pa", "minsan", "dito", "doon", "saan", "lahat", "anumang", "kapwa", "bawat", "ilan",
    "karamihan", "iba", "tulad", "lamang", "pareho", "kaya", "kaysa", "masyado", "napaka", "isa",
    "bababa", "kulang", "marami", "ngayon", "kailanman", "sabi", "nabanggit", "din", "kumuha",
    "pumunta", "pumupunta", "ilagay", "makita", "nakita", "katulad", "mahusay", "likod", "kahit",
    "paraan", "noon", "gayunman", "dalawa", "tatlo", "apat", "lima", "una", "pangalawa"
]

def preprocess_text(text):
    text = str(text)
    doc = nlp(text)
    tokens = [
        token.text.lower() for token in doc
        if not token.is_punct and not token.is_space and token.text.isalpha()
        and token.text.lower() not in tagalog_stopwords
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
