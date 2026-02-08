# 🏦 Financial Consumer Complaints Classification 

## 📋 Project Overview
This project applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify real-world consumer complaints into financial product categories (e.g., Mortgages, Credit Cards, Debt Collection).

Leveraging the **US Consumer Complaint Database**, the model processes over **330,000 anonymized narratives** to automate customer support routing and trend analysis.

## 🎯 Key Features
* **Real-World Data:** Transitioned from synthetic data to a massive dataset of 330k+ records.
* **Production-Grade Preprocessing:** Custom pipeline handling anonymization masks (`XXXX`), legal jargon, and text normalization using **SpaCy**.
* **Scalable Architecture:** Modularized code (`src/`) separating data engineering from modeling logic.
* **Class Imbalance Handling:** Strategies to manage dominant categories like Credit Reporting vs. smaller ones like Retail Banking.

### 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Libraries:** Pandas, Scikit-learn, NLTK, Spacy, Streamlit.
* **Visualization:** Matplotlib, Seaborn, Power BI.
* **Dataset:** US Consumer Complaint Database (Kaggle).
