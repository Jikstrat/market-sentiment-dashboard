import pandas as pd
import yfinance as yf
from tqdm import tqdm

INPUT_FILE = "sentiment_system/data/feature_dataset.csv"
OUTPUT_FILE = "sentiment_system/data/training_dataset.csv"


def get_stock_prices(symbol):

    ticker = symbol + ".NS"

    data = yf.download(
        ticker,
        period="1y",
        progress=False
    )

    data.reset_index(inplace=True)

    # Handle possible multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data[["Date", "Close"]]


def main():

    print("Loading feature dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    symbols = df["symbol"].unique()

    price_data = {}

    print("Downloading stock price data...")

    for symbol in tqdm(symbols):

        prices = get_stock_prices(symbol)

        prices["Date"] = pd.to_datetime(prices["Date"])

        price_data[symbol] = prices

    directions = []

    print("Labeling price movements...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        symbol = row["symbol"]
        news_date = row["date"]

        prices = price_data[symbol].copy()

        prices = prices.sort_values("Date")

        # Find first trading day >= news day
        future_prices = prices[prices["Date"] >= news_date]

        if len(future_prices) < 2:
            directions.append(None)
            continue

        today_price = future_prices.iloc[0]["Close"]
        next_price = future_prices.iloc[1]["Close"]

        # Ensure numeric
        today_price = float(today_price)
        next_price = float(next_price)

        if next_price > today_price:
            directions.append("UP")
        else:
            directions.append("DOWN")

    df["direction"] = directions

    df = df.dropna()

    print("Final training dataset size:", len(df))

    df.to_csv(OUTPUT_FILE, index=False)

    print("Training dataset saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()