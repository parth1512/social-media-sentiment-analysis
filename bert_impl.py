from transformers import pipeline

def analyze_sentiment_bert(text_list):
    # Initialize the sentiment analysis pipeline
    sentiment_model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    
    # Process text
    results = sentiment_model(text_list)
    return results

if __name__ == "__main__":
    # Example usage for testing
    import pandas as pd
    df = pd.DataFrame({'text': ["I love AI", "This is bad"]})
    df['bert_sentiment'] = df['text'].apply(lambda text: analyze_sentiment_bert([text])[0]['label'])
    print(df[['text', 'bert_sentiment']])
