import streamlit as st
import pandas as pd
import yfinance as yf

from market_ranker import rank_market
from sentiment_system.predictor import predict_stock

STOCK_FILE = "sentiment_system/data/nifty50_stocks.csv"


def get_stock_chart(symbol):

    ticker = symbol + ".NS"

    data = yf.download(
        ticker,
        period="3mo",
        interval="1d",
        progress=False
    )

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    chart_data = data[["Date", "Close"]]

    chart_data = chart_data.set_index("Date")

    return chart_data


st.set_page_config(
    page_title="Market Sentiment Dashboard",
    layout="wide"
)

st.title("Market Sentiment Dashboard")

st.caption("News-driven market analysis for NIFTY 50 equities")

stocks = pd.read_csv(STOCK_FILE)

# ------------------------------
# STOCK ANALYSIS

st.subheader("Stock Analysis")

stock_symbols = stocks["symbol"].tolist()

selected_stock = st.selectbox("Select Stock", stock_symbols)

company = stocks[stocks["symbol"] == selected_stock]["company"].values[0]

if st.button("Analyze Stock"):

    with st.spinner("Analyzing latest market signals..."):
        result = predict_stock(selected_stock, company)

    if result:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Market Outlook", result["prediction"])

        with col2:
            st.metric("Upside Probability", round(result["up_prob"], 3))

        with col3:
            st.metric("Downside Probability", round(result["down_prob"], 3))

        st.divider()

        st.subheader("Relevant Market Headlines")

        for headline in result["news"]:
            st.write("•", headline)

        st.divider()

        st.subheader("Price Trend (3 Months)")

        price_data = get_stock_chart(selected_stock)

        if price_data is not None:
            st.line_chart(price_data)
        else:
            st.warning("Price data unavailable.")

    else:
        st.warning("No recent market news available for this stock.")

st.divider()

# ------------------------------
# MARKET SENTIMENT

st.subheader("Market Sentiment Overview")

if st.button("Generate Market Overview"):

    with st.spinner("Scanning market sentiment..."):
        bullish, bearish = rank_market()

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Top Positive Sentiment Stocks**")

        if not bullish.empty:
            st.dataframe(bullish[["symbol", "up_prob"]])
        else:
            st.write("No strong positive sentiment detected.")

    with col2:

        st.markdown("**Top Negative Sentiment Stocks**")

        if not bearish.empty:
            st.dataframe(bearish[["symbol", "down_prob"]])
        else:
            st.write("No strong negative sentiment detected.")

st.divider()

# ------------------------------
# SYSTEM DETAILS

st.subheader("System Details")

st.write("This dashboard evaluates market sentiment signals derived from financial news and historical price behaviour.")

col1, col2 = st.columns(2)

with col1:

    st.markdown("**Analytical Model**")

    st.write("Random Forest Classifier")

    st.markdown("**Training Dataset Size**")

    st.write("3924 observations")

with col2:

    st.markdown("**Directional Accuracy**")

    st.write("~56%")

st.caption("Data sources include financial news feeds and historical market data.")