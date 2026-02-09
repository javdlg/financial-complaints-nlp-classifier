import spacy
import re
import pandas as pd

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    raise RuntimeError(
        "❌ Model not found. Please run: python -m spacy download en_core_web_sm"
    )


def clean_text(text):
    """
    Preprocesses text specifically for the Consumer Complaint Database.

    Pipeline:
    1. Lowercasing.
    2. Anonymization mask removal ('XXXX').
    3. URL & HTML removal.
    4. Non-alphabetic character removal.
    5. SpaCy Lemmatization & Stopword removal.
    """
    if pd.isna(text) or text == "":
        return ""

    # 1. Lowercase
    text = str(text).lower()

    # 2. Remove anonymization masks (Specific to this dataset)
    # The dataset uses 'XXXX' to hide personal info. We remove these patterns.
    text = re.sub(r"x{2,}", "", text)  # Removes 'xx', 'xxx', 'xxxx'

    # 3. Remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)

    # 4. Remove non-alphabetic characters (numbers, punctuation)
    # We remove numbers because dates/amounts are usually anonymized or irrelevant for sentiment/topic
    text = re.sub(r"[^a-z\s]", "", text)

    # 5. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. SpaCy Processing
    doc = nlp(text)

    # Custom stopwords for banking context
    # We keep it minimal initially to see what the model learns
    custom_stopwords = {"ve", "don", "ll", "did", "does"}

    clean_tokens = []
    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and len(token.text) > 1
            and token.text not in custom_stopwords
        ):
            clean_tokens.append(token.lemma_)

    return " ".join(clean_tokens)
