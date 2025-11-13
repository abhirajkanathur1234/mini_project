import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    .fake-job {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .real-job {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .feature-badge {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px;
        border-radius: 15px;
        font-size: 0.8rem;
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .safety-tips {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model and vectorizer"""
    try:
        # You'll need to replace these with your actual model file paths
        with open('models/rf_adasyn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the model files are in the correct location.")
        return None, None

def preprocess_input_text(text):
    """Preprocess input text using the same preprocessing as training"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    # Join tokens back to string
    return ' '.join(tokens)

def predict_job_authenticity(text, model, vectorizer, threshold=0.3):
    """Predict if a job posting is authentic or fake"""
    # Preprocess the text
    processed_text = preprocess_input_text(text)
    
    # Transform using TF-IDF
    text_tfidf = vectorizer.transform([processed_text])
    
    # Get prediction and probability
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0][1]  # Probability of being fake
    
    # Apply threshold
    final_prediction = 1 if probability > threshold else 0
    
    return final_prediction, probability, processed_text

def main():
    st.markdown('<h1 class="main-header">🔍 Fake Job Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Random Forest with ADASYN")
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None:
        st.warning("Please make sure your model files are in the 'models/' directory")
        return
    
    # Create single tab interface
    st.header("Analyze Job Posting")
    
    with st.form("job_analysis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("Job Title*", placeholder="e.g., Senior Software Engineer")
            company_profile = st.text_area("Company Profile*", placeholder="Describe the company...", height=100)
        
        with col2:
            job_description = st.text_area("Job Description*", placeholder="Describe the job responsibilities...", height=150)
            requirements = st.text_area("Requirements*", placeholder="List the requirements...", height=100)
        
        submitted = st.form_submit_button("🔍 Analyze Job Posting")
    
    if submitted:
        if not all([job_title, company_profile, job_description, requirements]):
            st.warning("Please fill in all required fields.")
        else:
            combined_text = f"{job_title} {company_profile} {job_description} {requirements}"
            
            with st.spinner("Analyzing job posting..."):
                prediction, probability, processed_text = predict_job_authenticity(
                    combined_text, model, vectorizer
                )
            
            # Display results
            st.markdown("---")
            st.subheader("Analysis Results")
            
            if prediction == 1:
                st.markdown(f'<div class="result-box fake-job">', unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.error("🚩 **FAKE JOB DETECTED**")
                with col2:
                    st.metric("Confidence", f"{probability:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box real-job">', unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.success("✅ **REAL JOB**")
                with col2:
                    st.metric("Confidence", f"{probability:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk level
            if probability > 0.7:
                risk_level = "HIGH RISK"
                risk_color = "red"
            elif probability > 0.4:
                risk_level = "MEDIUM RISK"
                risk_color = "orange"
            else:
                risk_level = "LOW RISK"
                risk_color = "green"
            
            st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")
            
            # Key features
            st.subheader("Key Factors Considered")
            feature_names = vectorizer.get_feature_names_out()
            text_tfidf = vectorizer.transform([processed_text])
            feature_values = text_tfidf.toarray()[0]
            
            # Get top features
            non_zero_indices = np.where(feature_values > 0)[0]
            if len(non_zero_indices) > 0:
                important_features = []
                for idx in non_zero_indices:
                    if idx < len(model.feature_importances_):
                        importance = model.feature_importances_[idx]
                        important_features.append((feature_names[idx], feature_values[idx], importance))
                
                important_features.sort(key=lambda x: x[2], reverse=True)
                
                st.write("Top influential terms detected:")
                cols = st.columns(3)
                for i, (feature, value, importance) in enumerate(important_features[:6]):
                    with cols[i % 3]:
                        st.markdown(f'<div class="feature-badge">{feature}</div>', unsafe_allow_html=True)
            
            # Safety Tips
            st.markdown("---")
            st.markdown('<div class="safety-tips">', unsafe_allow_html=True)
            st.subheader("💡 SAFETY TIPS")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ DO:**")
                st.write("• Research companies thoroughly")
                st.write("• Verify official contact information")
                st.write("• Use professional networking sites")
                st.write("• Trust your instincts")
                st.write("• Check company reviews and ratings")
                st.write("• Verify through official company websites")
            
            with col2:
                st.write("**❌ DON'T:**")
                st.write("• Pay upfront fees or deposits")
                st.write("• Share sensitive personal information early")
                st.write("• Accept vague job descriptions")
                st.write("• Rush into decisions")
                st.write("• Share bank account details")
                st.write("• Send money for 'training materials'")
            
            # Additional warnings based on prediction
            if prediction == 1:
                st.warning("**🚨 EXTRA CAUTION ADVISED:**")
                st.write("• This job shows strong characteristics of fraudulent postings")
                st.write("• Verify the company through multiple independent sources")
                st.write("• Be extremely cautious of any payment requests")
                st.write("• Report suspicious postings to the platform")
            else:
                st.info("**📝 STANDARD PRECAUTIONS:**")
                st.write("• Continue with standard application procedures")
                st.write("• Research the company independently")
                st.write("• Verify job details through official channels")
                st.write("• Be cautious during the interview process")
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()