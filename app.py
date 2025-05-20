from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import calamancy
from typing import List, Dict, Tuple
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

# Load stopwords
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

# Load calamancy model
nlp = calamancy.load("tl_calamancy_md-0.1.0")

# Load ML model & tokenizer
svm_model = joblib.load("svm_model.pkl")
feature_means = np.load("svm_feature_means.npy")

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize FastAPI
app = FastAPI(
    title="SMS Spam Detection API",
    description="API for detecting spam messages in SMS using machine learning",
    version="1.0.0"
)

# Input schema
class SMSRequest(BaseModel):
    message: str
    sender: str

# Preprocessing
def preprocess_text(text: str) -> str:
    text = str(text)
    doc = nlp(text)
    tokens = [
      token.text.lower() for token in doc
      if not token.is_punct and not token.is_space and token.text.isalpha()
      and token.text.lower() not in tagalog_stopwords
    ]
    return " ".join(tokens) if tokens else text.lower()

# Embedding extraction
def get_embeddings(text: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        output = model(**encoded)
        cls_embedding = output.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# Sender feature engineering
def is_numeric(s: str) -> int:
    return int(str(s).isdigit())

def is_short_code(s: str) -> int:
    return int(len(str(s)) < 6)

# Word importance analysis
def analyze_word_importance(text: str, embedding: np.ndarray, prediction: int) -> List[Tuple[str, float]]:
    # Use the same preprocessing as training
    doc = nlp(text)
    # print("Original tokens and lemmas:")
    # for token in doc:
    #   print(f"Token: '{token.text}', Lemma: '{token.lemma_}'")

    words = [
      token.text.lower() for token in doc
      if not token.is_punct and not token.is_space and token.text.isalpha()
      and token.text.lower() not in tagalog_stopwords
    ]

    print("Words after preprocessing:", words)

    # Get word embeddings
    word_embeddings = {}
    for word in set(words):
        word_embedding = get_embeddings(word)
        word_embeddings[word] = word_embedding

    # Calculate importance scores
    importance_scores = []
    for word in words:
        if word in word_embeddings:
            sim = np.dot(word_embeddings[word], embedding) / (
                np.linalg.norm(word_embeddings[word]) * np.linalg.norm(embedding)
            )
            importance_scores.append((word, sim if prediction == 1 else -sim))

    # Sort by absolute importance
    importance_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return importance_scores  # Return all words


# Visualization generator
def create_visualization(text: str, sender: str, embedding: np.ndarray, prediction: int, confidence: float, processed: str) -> str:
    plt.figure(figsize=(20, 15))

    # 1. Message Length
    plt.subplot(3, 2, 1)
    plt.bar(['Message Length'], [len(text)], color='skyblue')
    plt.title('Message Length')
    plt.ylabel('Characters')

    # 2. Sender Type
    plt.subplot(3, 2, 2)
    sender_types = ['Numeric', 'Short Code']
    sender_values = [is_numeric(sender), is_short_code(sender)]
    plt.bar(sender_types, sender_values, color=['lightgreen', 'lightcoral'])
    plt.title('Sender Characteristics')
    plt.ylim(0, 1)

    # 3. Embedding slice
    plt.subplot(3, 2, 3)
    plt.plot(embedding[:10], marker='o')
    plt.title('First 10 Embedding Dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Value')

    # 4. Confidence
    plt.subplot(3, 2, 4)
    plt.bar(['Spam', 'Ham'], [confidence, 1-confidence],
            color=['red' if prediction == 1 else 'green',
                  'green' if prediction == 1 else 'red'])
    plt.title('Prediction Confidence')
    plt.ylim(0, 1)

    # 5. Important Words
    plt.subplot(3, 2, 5)
    word_importance = analyze_word_importance(processed, embedding, prediction)
    if word_importance:  # Check if we have any words
        words, scores = zip(*word_importance)
        colors = ['red' if score > 0 else 'green' for score in scores]
        plt.barh(words, [abs(score) for score in scores], color=colors)
        plt.title('Word Importance')
    else:
        plt.text(0.5, 0.5, 'No significant words found',
                horizontalalignment='center', verticalalignment='center')
        plt.title('Word Importance')
    plt.xlabel('Importance Score')

    # 6. Highlighted Text
    plt.subplot(3, 2, 6)
    plt.axis('off')
    highlighted = text
    if word_importance:
      for word, score in word_importance:
          color = 'red' if score > 0 else 'green'
          highlighted = re.sub(rf'\b{re.escape(word)}\b', f'[{word.upper()}]', highlighted, flags=re.IGNORECASE)
    plt.text(0.1, 0.5, highlighted, wrap=True, fontsize=10)
    plt.title('Highlighted Text (Red=Spam, Green=Ham)')


    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    return base64.b64encode(image_png).decode()

# Prediction endpoint
@app.post("/predict")
def classify_sms(request: SMSRequest) -> Dict:
    # Preprocess text
    processed = preprocess_text(request.message)

    # Get embeddings
    embedding = get_embeddings(processed)

    # Get sender features
    sender_features = np.array([
        is_numeric(request.sender),
        is_short_code(request.sender)
    ])

    # Combine features
    features = np.hstack([embedding, sender_features])

    # Make prediction
    prediction = svm_model.predict([features])[0]

    # Get confidence scores
    confidence_score = svm_model.decision_function([features])[0]
    confidence = 1 / (1 + np.exp(-confidence_score))

    # Create visualization (pass processed)
    vis = create_visualization(
        request.message,
        request.sender,
        embedding,
        prediction,
        confidence,
        processed
    )

    # Get word importance (use processed)
    importance = analyze_word_importance(processed, embedding, prediction)

    return {
        "text": request.message,
        "sender": request.sender,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": float(confidence),
        "embedding": embedding.tolist(),
        "important_words": [
            {"word": word, "importance": float(score)}
            for word, score in importance
        ],
        "visualization": vis
    }

# Add root endpoint for health check
@app.get("/")
def root():
    return {"status": "healthy", "message": "SMS Spam Detection API is running"}