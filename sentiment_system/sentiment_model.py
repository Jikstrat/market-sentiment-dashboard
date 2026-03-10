import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

INPUT_FILE = "sentiment_system/data/raw_news.csv"
OUTPUT_FILE = "sentiment_system/data/sentiment_news.csv"

MODEL_NAME = "ProsusAI/finbert"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    return tokenizer, model


def predict_sentiment(text, tokenizer, model):

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    score, label = torch.max(probs, dim=1)

    labels = ["negative", "neutral", "positive"]

    return labels[label], score.item()


def main():

    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    tokenizer, model = load_model()

    sentiments = []
    scores = []

    print("Running FinBERT sentiment analysis...")

    for headline in tqdm(df["clean_headline"]):

        label, score = predict_sentiment(headline, tokenizer, model)

        sentiments.append(label)
        scores.append(score)

    df["sentiment_label"] = sentiments
    df["sentiment_score"] = scores

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved sentiment dataset to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()