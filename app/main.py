import streamlit as st
import joblib
import pandas as pd
import sys
import os

# ---------------------------------------------------------
# 1. PATH SETUP & IMPORTS
# ---------------------------------------------------------
# We need to add the parent directory to sys.path to import 'src'
# This allows us to reuse the exact cleaning logic from training.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.preprocessing import clean_text
except ImportError:
    st.error("❌ Could not import 'src.preprocessing'. Make sure you run this from the project root.")
    st.stop()

# ---------------------------------------------------------
# 2. CONFIGURATION & CACHING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Financial Complaint Classifier",
    page_icon="🏦",
    layout="centered"
)

# Load models only once using streamlit's cache
@st.cache_resource
def load_model_pipeline():
    try:
        # Paths relative to the app/ folder
        model_path = os.path.join(parent_dir, 'models', 'random_forest_classifier.pkl')
        vectorizer_path = os.path.join(parent_dir, 'models', 'tfidf_vectorizer.pkl')
        
        # Verify files exist before loading
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return None, None
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load the pipeline
model, vectorizer = load_model_pipeline()

# ---------------------------------------------------------
# 3. UI LAYOUT
# ---------------------------------------------------------
st.title("🏦 Financial Consumer Complaint Classifier")
st.markdown("""
**Analyze and route customer complaints automatically.** This NLP tool classifies US consumer finance complaints into 5 standard banking categories:
* 🏠 **Mortgages**
* 💳 **Credit Cards**
* 📉 **Debt Collection**
* 🏦 **Retail Banking**
* 📊 **Credit Reporting**
""")

st.info("💡 **Tip:** Try pasting a real complaint about a late fee, a foreclosure, or a credit report error.")

# Check if models loaded correctly
if model is None or vectorizer is None:
    st.error("⚠️ Models not found! Please check if '.pkl' files exist in the 'models/' folder.")
    st.stop()

# Input Area
user_input = st.text_area(
    "Enter complaint narrative:", 
    height=150,
    placeholder="Example: I have been trying to contact the bank regarding a fraudulent charge on my card but they keep ignoring me..."
)

# ---------------------------------------------------------
# 4. PREDICTION LOGIC
# ---------------------------------------------------------
if st.button("🔍 Classify complaint", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Processing text and analyzing patterns..."):
            # A. Preprocessing (The exact same function used in training)
            cleaned_text = clean_text(user_input)
            
            # Debug Expander (Good for demos)
            with st.expander("See how the model 'reads' this text (Preprocessing)"):
                st.text(f"Raw: {user_input}")
                st.text(f"Cleaned (Lemmatized): {cleaned_text}")
            
            if len(cleaned_text) < 2:
                st.warning("The text is too short or contains only stopwords/numbers after cleaning. Please provide more detail.")
            else:
                # B. Vectorization
                input_vector = vectorizer.transform([cleaned_text])
                
                # C. Prediction
                prediction = model.predict(input_vector)[0]
                probabilities = model.predict_proba(input_vector)[0]
                
                # D. Display results
                st.success(f"### 🎯 Prediction: **{prediction}**")
                
                # E. Probability chart
                st.subheader("Confidence levels")
                
                # Create DataFrame for the chart
                classes = model.classes_
                prob_df = pd.DataFrame({
                    'Category': classes,
                    'Probability': probabilities
                })
                
                # Highlight the winner
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                
                # Use streamlit's native bar chart
                st.bar_chart(
                    prob_df.set_index('Category'),
                    color="#4CAF50" # Green color for bars
                )
                
# Footer
st.markdown("---")
st.caption("Built with Python, Scikit-Learn, SpaCy & Streamlit. Model trained on US Consumer Complaint Database.")