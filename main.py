from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
import os

app = FastAPI(
    title="Capstone Model API",
    description="API for the Capstone Model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load spaCy model
try:
    nlp = spacy.load("tl_core_news_md")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download tl_core_news_md")
    nlp = spacy.load("tl_core_news_md")

@app.get("/")
async def root():
    return {"message": "Welcome to Capstone Model API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add your model endpoints here 