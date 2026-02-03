import spacy
import re
import pandas as pd

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    # Fallback to prevent immediate crash if not installed
    nlp = None

def clean_text(text):
    """
    Applies a complete NLP preprocessing pipeline:
    1. Lowercasing
    2. URL and HTML removal
    3. Special character removal
    4. Tokenization and Lemmatization (using spaCy)
    5. Stopwords removal
    """
    if pd.isna(text) or text == "":
        return ""
    
    # 1. Lowercase
    text = str(text).lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S', '', text, flags=re.MULTILINE)

    # 3. Remove HTML tags 
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove special characters and numbers (keep only letters)
    text = re.sub(r'[a-z\s]', '', text)

    # If spaCy is not loaded, return basic cleaned text (safety check)
    if nlp is None:
        return text.strip()
    
    # SpaCy pipeline: Tokenization -> Lemmatization -> Stopwords
    doc = nlp(text)

    # Keep tokens that are NOT stopwords, punctuation and longer than 2 characters
    clean_tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]

    return " ".join(clean_tokens)