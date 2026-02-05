from pydoc import doc
import spacy
import re
import pandas as pd

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded successfully.")
except OSError:
    print("❌ SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
    print("⚠️ Solution: Execute in your terminal: python -m spacy download en_core_web_sm")
    # Throw an specific error for stopping the notebook execution
    raise RuntimeError("SpaCy model 'en_core_web_sm' is not installed.")
    

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

    # 4. Remove special characters but keep numbers and letters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # SpaCy pipeline: Tokenization -> Lemmatization -> Stopwords
    doc = nlp(text)

    clean_tokens = []
    for token in doc:
        # Relaxed logic:
        # - Is not a stopword
        # - Is not punctuation
        # - Lenght > 1 (Save 2 letter words as 'pc', 'it', 'ui')
        if not token.is_stop and not token.is_punct and len(token.text) > 1:
            clean_tokens.append(token.lemma_)

    return " ".join(clean_tokens)