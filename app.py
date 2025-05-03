from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch

from transformers import AutoTokenizer, AutoModel
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy

app = FastAPI()

# Load NLP tools

nlp = spacy.load("tl_calamancy_md")



tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
transformer_model = AutoModel.from_pretrained("xlm-roberta-base")
svm_model = joblib.load("svm_model.pkl")

# Stopwords
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
        token.lemma_.lower() for token in doc
        if not token.is_punct and not token.is_space and token.lemma_.isalpha()
        and token.lemma_.lower() not in tagalog_stopwords
    ]
    return " ".join(tokens) if tokens else text.lower()


def get_embeddings(text_list):
    transformer_model.eval()
    with torch.no_grad():
        encoded = tokenizer(text_list, return_tensors='pt', truncation=True, padding=True, max_length=128)
        output = transformer_model(**encoded)
        cls_embeddings = output.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        return cls_embeddings.numpy()


class SMSRequest(BaseModel):
    message: str

@app.post("/predict")
def predict(data: SMSRequest):
    processed = preprocess_text(data.message)
    print(f"🧼 Processed Text: {processed}")

    embedding = get_embeddings([processed])
    print(f"🔢 Embedding Shape: {embedding.shape}")
    print(f"📊 Embedding Sample: {embedding[0][:10]}")  # first 10 values

    prediction = svm_model.predict(embedding)[0]
    return {
        "prediction": "Spam" if prediction == 1 else "Ham",
        "processed": processed,
        "embedding": embedding.tolist()  # convert to list for JSON serialization
    }