# Social Media Sentiment Analysis & Bot Detection

## 📌 Overview
This project is a comprehensive tool for analyzing social media data (specifically Twitter/X). It provides two core functionalities:
1.  **Bot Detection**: A Machine Learning model (Random Forest) that detects whether a tweet was written by a Human or a Bot.
2.  **Sentiment Analysis**: A Deep Learning pipeline (DistilBERT) that classifies text as Positive, Negative, or Neutral.

The project includes a **Streamlit** frontend for easy interaction with these models.

## 🚀 Features
-   **Interactive Web UI**: Built with Streamlit for easy model training and testing.
-   **Machine Learning**: Random Forest Classifier for bot detection (~85% accuracy on balanced datasets).
-   **Deep Learning**: Hugging Face Transformers (DistilBERT) for state-of-the-art sentiment analysis.
-   **Data visualization**: Confusion matrices and sentiment distribution charts.
-   **Modular Design**: Refactored Python scripts for better maintainability.

## 🛠️ Installation

### Prerequisites
-   Python 3.8+
-   pip

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/parth1512/social-media-sentiment-analysis.git
    cd social-media-sentiment-analysis
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `requirements.txt` is missing, install manually)*:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn nltk textblob tweepy transformers streamlit
    ```

## 🖥️ Usage

### Running the Web App
To start the user interface:
```bash
streamlit run app.py
```
This will open the app in your browser at `http://localhost:8501`.

### Modules
-   **Bot Detection**: Navigate to the "Bot Detection" tab, click "Train Model", and then enter any tweet to check if it's from a bot.
-   **Sentiment Analysis**: Navigate to the "Sentiment Analysis" tab and enter text to see if it's Positive or Negative.

## 📂 Project Structure
-   `app.py`: Main Streamlit application entry point.
-   `random_forest_algo.py`: Logic for training the Bot Detection model.
-   `bert_impl.py`: Logic for BERT-based sentiment analysis.
-   `api_setup.py`: Script to fetch tweets using Twitter API (requires API keys).
-   `data/`: Contains datasets (e.g., `bot_detection_data.csv`).

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.