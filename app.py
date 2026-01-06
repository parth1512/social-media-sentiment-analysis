
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Import our refactored modules
# Note: We need to ensure the venv is active and path is correct. 
# Streamlit should be run from the root directory.
import random_forest_algo as rf
import bert_impl as bert

# Set page config
st.set_page_config(
    page_title="Social Media Analytics",
    page_icon="🐦",
    layout="wide"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DA1F2;
        color: white;
    }
    h1 {
        color: #1DA1F2;
    }
</style>
""", unsafe_allow_html=True)

# Application Title
st.title("🐦 Social Media Sentiment & Bot Detection")

# Sidebar Navigation
mode = st.sidebar.radio("Select Mode", ["Bot Detection", "Sentiment Analysis"])

if mode == "Bot Detection":
    st.header("🤖 Bot Detection System")
    st.markdown("Train a Random Forest model to distinguish between Human and Bot tweets.")

    # Session State for Model
    if 'rf_model' not in st.session_state:
        st.session_state.rf_model = None
        st.session_state.vectorizer = None

    # Train Button
    if st.button("Train Model (using bot_detection_data.csv)"):
        with st.spinner("Training Model... Please wait description"):
            model, vec, report, conf_matrix = rf.load_and_train_model()
            
            if isinstance(model, str): # Error message
                st.error(model)
            else:
                st.session_state.rf_model = model
                st.session_state.vectorizer = vec
                st.session_state.report = report
                st.session_state.conf_matrix = conf_matrix
                st.success("Model Trained Successfully!")

    # Display Metrics if Trained
    if st.session_state.rf_model:
        st.subheader("Model Performance")
        
        # Metrics Columns
        c1, c2, c3 = st.columns(3)
        accuracy = st.session_state.report['accuracy']
        precision = st.session_state.report['1']['precision'] # Bot class precision
        recall = st.session_state.report['1']['recall'] # Bot class recall
        
        c1.metric("Accuracy", f"{accuracy:.2%}")
        c2.metric("Bot Precision", f"{precision:.2%}")
        c3.metric("Bot Recall", f"{recall:.2%}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(st.session_state.conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Interactive Prediction
        st.subheader("Test the Model")
        user_input = st.text_input("Enter a tweet to classify:")
        if user_input:
            prediction = rf.predict_single(user_input, st.session_state.rf_model, st.session_state.vectorizer)
            if prediction == "Bot":
                st.error(f"🤖 Prediction: **{prediction}**")
            else:
                st.success(f"👤 Prediction: **Human**")

elif mode == "Sentiment Analysis":
    st.header("❤️ Sentiment Analysis (BERT)")
    st.markdown("Analyze the sentiment of text using a pre-trained DistilBERT model.")

    # Input Area
    text_input = st.text_area("Enter text to analyze (for multiple tweets, separate by new lines):", height=150)
    
    if st.button("Analyze Sentiment"):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                # Split by newlines for multiple inputs
                texts = [t.strip() for t in text_input.split('\n') if t.strip()]
                
                if texts:
                    results = bert.analyze_sentiment_bert(texts)
                    
                    # Display Results
                    st.subheader("Results")
                    
                    # Prepare Data for Chart
                    labels = [r['label'] for r in results]
                    scores = [r['score'] for r in results]
                    
                    # Display individual results
                    for t, r in zip(texts, results):
                        emoji = "🟢" if r['label'] == 'POSITIVE' else "🔴"
                        st.write(f"{emoji} **{r['label']}** ({r['score']:.4f}): _{t}_")

                    # Chart
                    if len(results) > 1:
                        st.subheader("Sentiment Distribution")
                        df_res = pd.DataFrame(labels, columns=["Sentiment"])
                        
                        fig, ax = plt.subplots()
                        df_res['Sentiment'].value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
                        plt.title("Sentiment Counts")
                        st.pyplot(fig)
                else:
                    st.warning("Please enter valid text.")
        else:
             st.warning("Please enter some text.")

st.sidebar.markdown("---")
st.sidebar.info("Social Media Sentiment Analysis Project")
