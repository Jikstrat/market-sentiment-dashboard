import pandas as pd
import joblib
import feedparser
import re

MODEL_PATH = "models/random_forest_model.pkl"


def clean_text(text):

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)

    return text.lower().strip()


def fetch_latest_news(company):

    query = company.replace(" ", "+") + "+stock"

    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    feed = feedparser.parse(url)

    headlines = []

    for entry in feed.entries[:10]:
        headlines.append(entry.title)

    return headlines


def generate_features(headlines):

    sentiment_strength = len(headlines) * 0.05

    rolling_3 = sentiment_strength * 0.8
    rolling_7 = sentiment_strength * 0.6

    news_count = len(headlines)

    return [[sentiment_strength, rolling_3, rolling_7, news_count]]


def predict_stock(symbol, company):

    model = joblib.load(MODEL_PATH)

    headlines = fetch_latest_news(company)

    if len(headlines) == 0:
        return None

    features = generate_features(headlines)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    return {
        "symbol": symbol,
        "prediction": prediction,
        "up_prob": probabilities[1],
        "down_prob": probabilities[0],
        "news": headlines
    }