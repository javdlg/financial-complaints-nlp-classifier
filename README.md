# 🎫 Intelligent Customer Support Ticket Routing System

### 📋 Project Overview
This project leverages **Natural Language Processing (NLP)** to automate the classification and prioritization of customer support tickets. By analyzing unstructured text data from customer queries, the system predicts the appropriate **department** (e.g., Billing, Technical Support) and assigns a **priority level**, aiming to reduce response times and optimize support team workflows.

### 🎯 Key Objectives
* **Automate Routing:** Reduce manual triage by accurately classifying tickets.
* **NLP Pipeline:** Implement a robust text processing pipeline including tokenization, lemmatization, and TF-IDF vectorization.
* **Business Intelligence:** Visualize ticket trends and model performance using **Power BI**.
* **Deployment:** Interactive demo using **Streamlit** for real-time classification.

### 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Libraries:** Pandas, Scikit-learn, NLTK, Spacy, Streamlit.
* **Visualization:** Matplotlib, Seaborn, Power BI.
* **Dataset:** Customer Support Ticket Dataset (Kaggle).

---

### 🏠 Project Structure:

├── .gitignore             # Ignore venv, __pycache__, datasets/
├── README.md              
├── requirements.txt       # Dependencies (pandas, scikit-learn, nltk, spacy, streamlit)
├── data/
│   ├── raw/               # Original dataset from Kaggle
│   ├── processed/         # Clean data and ready for the model
│   └── external/          # stop-words dictionary or extra lemas
├── notebooks/             # Where the magic happens
│   ├── EDA.ipynb
│   ├── preprocessing-normalization.ipynb
│   └── modeling-evaluation.ipynb
├── src/                   # Modularized code (for production)
│   ├── __init__.py
│   ├── preprocessing.py   # Cleaning functions (reusables)
│   └── visualization.py   # Customized graphics
├── app/                   # Interactive demo
│   └── main.py            # Streamlit script
└── reports/               
    └── project_report.pdf