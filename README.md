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

.
├── .gitignore               # Files and folders to be ignored by Git
├── README.md                # Project documentation and overview
├── requirements.txt         # Project dependencies and libraries
├── data/
│   ├── raw/                 # Original Kaggle dataset (immutable)
│   ├── processed/           # Cleaned and preprocessed data for modeling
│   └── external/            # Additional resources (dictionaries, etc.)
├── notebooks/               # Jupyter notebooks for experimentation
│   ├── 1_EDA_Exploratory.ipynb
│   ├── 2_Preprocessing_Normalization.ipynb
│   └── 3_Modeling_Evaluation.ipynb
├── src/                     # Modular Python scripts for production
│   ├── __init__.py
│   ├── preprocessing.py     # Text cleaning and normalization functions
│   └── visualization.py     # Custom plotting functions for reports
├── app/                     # Web application (Demo)
│   └── main.py              # Streamlit dashboard script
└── reports/                 # Project insights and academic documentation
    └── project_report.pdf   # Final technical report