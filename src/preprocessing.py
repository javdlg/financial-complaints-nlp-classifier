from pydoc import doc
import spacy
import re
import pandas as pd

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded successfully.")
except OSError:
    print(
        "❌ SpaCy model not found. Please run: python -m spacy download en_core_web_sm"
    )
    print(
        "⚠️ Solution: Execute in your terminal: python -m spacy download en_core_web_sm"
    )
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

    # 2. Space normalization (remove extra spaces)
    text = re.sub(r"\s+", " ", text).strip()

    # 3. Remove URLs and HTML tags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)

    # 4. Remove special characters but keep letters
    text = re.sub(r"[^a-z\s]", "", text)

    # 5. Specific phrase removals based on dataset analysis
    text = text.replace("have issue productpurchased assist", "")
    text = text.replace("productpurchased", "")
    text = text.replace("productpurchase", "")  # Remove variations

    # SpaCy pipeline: Tokenization -> Lemmatization -> Stopwords
    doc = nlp(text)

    # Blacklist of specific thrash words of this dataset
    custom_stopwords = {
        "ve",
        "don",
        "ll",
        "assist",
        "issue",
        "have",
        "Having",
        "has",
        "had",
        "productpurchase",
        "productpurchased",
        "productprice",
        "productcost",
        "productid",
        "errormessage",
        "inplist",
        "threadinfo",
        "faqhelp",
        "faq",
    }

    clean_tokens = []
    for token in doc:
        # Filters: No stopwords, no punctuation, length > 1
        if (
            not token.is_stop
            and not token.is_punct
            and len(token.text) > 1
            and token.text not in custom_stopwords
            and token.lemma_ not in custom_stopwords
        ):
            clean_tokens.append(token.lemma_)

    return " ".join(clean_tokens)
