FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model and calamancy model
RUN python -m spacy download tl_core_news_lg
RUN python -m calamancy download tl_calamancy_md-0.1.0

# Copy model files
COPY svm_model.pkl .
# COPY svm_feature_means.npy .  # Uncomment if you're using feature means

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 