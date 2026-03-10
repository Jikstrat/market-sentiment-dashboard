import pandas as pd

INPUT_FILE = "sentiment_system/data/sentiment_news.csv"
OUTPUT_FILE = "sentiment_system/data/feature_dataset.csv"


def main():

    print("Loading sentiment dataset...")

    df = pd.read_csv(INPUT_FILE)

    # Ensure correct date format
    df["date"] = pd.to_datetime(df["date"])

    # Sort by company and date
    df = df.sort_values(["symbol", "date"])

    # Convert sentiment label → numeric
    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }

    df["sentiment_numeric"] = df["sentiment_label"].map(sentiment_map)

    # Sentiment strength
    df["sentiment_strength"] = abs(df["sentiment_score"] - 0.5)

    print("Generating rolling sentiment features...")

    # Rolling features per company
    df["rolling_3_sentiment"] = (
        df.groupby("symbol")["sentiment_numeric"]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_7_sentiment"] = (
        df.groupby("symbol")["sentiment_numeric"]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # News volume per day
    df["news_count_per_day"] = (
        df.groupby(["symbol", "date"])["headline"]
        .transform("count")
    )

    print("Cleaning dataset...")

    # Remove rows with insufficient rolling history
    df = df.dropna()

    print("Final dataset size:", len(df))

    # Safety check
    if len(df) < 300:
        raise ValueError("Dataset too small after feature engineering.")

    df.to_csv(OUTPUT_FILE, index=False)

    print("Feature dataset saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()