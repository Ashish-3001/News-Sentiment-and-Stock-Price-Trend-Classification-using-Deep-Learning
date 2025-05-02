import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import yfinance as yf
from tensorflow import keras
from tensorflow.keras import layers
from streamlit_option_menu import option_menu
from GoogleNews import GoogleNews
from datetime import datetime
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="S&P 100 Forecast", layout="wide")


st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)


selected = option_menu(
    menu_title=None,
    options=["Home", "Price Prediction", "Live Prediction", "News Sentiment"],
    icons=["house", "bar-chart-line", "graph-up-arrow", "newspaper"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#0e1117"},
        "icon": {"color": "white", "font-size": "20px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "color": "white",
            "padding": "10px 20px"
        },
        "nav-link-selected": {"background-color": "#262730"},
    }
)


@st.cache_resource
def load_model():
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        x = layers.Dropout(dropout)(x)
        return x + inputs + x

    def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)  
        for dim in mlp_units:
            x = layers.Dense(dim, activation="elu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        return keras.Model(inputs, outputs)

    model = build_model(
        input_shape=(10, 1),
        head_size=46,
        num_heads=60,
        ff_dim=55,
        num_transformer_blocks=1,
        mlp_units=[256],
        dropout=0.14,
        mlp_dropout=0.4,
    )
    model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["mean_squared_error"],
    )
    model.load_weights("transformer_model.weights.h5")
    return model

model = load_model()



@st.cache_resource
def load_sentiment_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    model.load_state_dict(torch.load("finbert_sentiment_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return tokenizer, model

tokenizer, sentiment_model = load_sentiment_model()

# Replace this block:
@st.cache_resource
def load_support_data():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("stockList.pkl", "rb") as f:
        stockList = pickle.load(f)
    return scaler, stockList

scaler, stockList = load_support_data()

def fetch_stock_data_yf(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    df.reset_index(inplace=True)
    df["Symbol"] = symbol
    return df[["Date", "Close", "Symbol"]].dropna()


def predict_sentiment(texts, return_probs=False):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
    labels = ["Negative", "Neutral", "Positive"]
    results = [(labels[p.item()], c.item()) for p, c in zip(preds, confs)]
    return results if return_probs else [r[0] for r in results]


def predict_for_stock_range(symbol, start_date, end_date, model, df_, scaler, window_size=10):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    if symbol not in df_:
        raise ValueError(f"Symbol '{symbol}' not found in the dataset.")

    stock_df = df_[symbol]
    custom_range = stock_df[(stock_df["Date"] >= start_date) & (stock_df["Date"] <= end_date)].reset_index(drop=True)
    
    if len(custom_range) <= window_size:
        raise ValueError(f"Not enough data in the range. Found {len(custom_range)} rows, need more than {window_size}.")

    sc = scaler[symbol]
    scaled_close = sc.transform(custom_range[["Close"]])

    X_input = []
    for i in range(window_size, len(scaled_close)):
        X_input.append(scaled_close[i - window_size:i, 0])
    X_input = np.array(X_input).reshape(-1, window_size, 1)

    y_pred_scaled = model.predict(X_input)
    y_pred = sc.inverse_transform(y_pred_scaled.reshape(-1, 1))

    result_df = custom_range.iloc[window_size:].copy()
    result_df["Predicted_Close"] = y_pred

    return result_df


if selected == "Home":
    st.markdown("<h1 style='text-align:center;'> S&P 100 Stock Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; font-size:18px; padding: 20px 40px;'>
    Welcome to the <b>S&P 100 Forecasting & Sentiment Dashboard</b> — an intelligent tool designed for <b>financial analysts, market researchers, and data-driven investors</b>.<br><br>
    This platform combines the power of a <b>Transformer-based deep learning model</b> for next-day stock price forecasting with real-time <b>news sentiment analysis</b> using state-of-the-art NLP models.<br><br>
    Whether you're backtesting historical predictions or seeking insights from the latest financial headlines, this dashboard empowers you to make informed decisions based on both <b>quantitative trends</b> and <b>qualitative sentiment</b>.<br><br>
    Use the navigation menu above to explore test set forecasts, get live predictions, or evaluate market sentiment for any S&P 100 stock.
    </div>
    """, unsafe_allow_html=True)


elif selected == "Price Prediction":
    st.title("Prediction on Custom Date Range")

    selected_stock = st.selectbox("Select a stock", stockList)

    default_start = datetime(2022, 1, 1)
    default_end = datetime.today()

    start_date = st.date_input("Start Date", default_start, min_value=datetime(2010, 1, 1), max_value=default_end)
    end_date = st.date_input("End Date", default_end, min_value=start_date, max_value=default_end)

    if st.button("Run Prediction"):
        try:
            st.info("Fetching data from Yahoo Finance...")
            raw_df = fetch_stock_data_yf(selected_stock, start_date, end_date)

            if raw_df.empty or len(raw_df) <= 10:
                st.warning("Not enough data available for prediction. Try a wider range.")
            else:
                raw_df["Date"] = pd.to_datetime(raw_df["Date"])

                result_df = predict_for_stock_range(
                    symbol=selected_stock,
                    start_date=start_date,
                    end_date=end_date,
                    model=model,
                    df_={selected_stock: raw_df},  
                    scaler=scaler,
                    window_size=10
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(result_df["Date"], result_df["Close"], label="Actual Close", linewidth=2)
                ax.plot(result_df["Date"], result_df["Predicted_Close"], label="Predicted Close", linestyle='--')
                ax.set_title(f"{selected_stock} Close Price Forecast")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price ($)")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                st.dataframe(result_df[["Date", "Close", "Predicted_Close"]])
                actual = result_df["Close"].values
                predicted = result_df["Predicted_Close"].values

                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, predicted)

                st.markdown("### Prediction Metrics")
                st.markdown(f"- **MAE (Mean Absolute Error):** `{mae:.2f}`")
                st.markdown(f"- **MSE (Mean Squared Error):** `{mse:.2f}`")
                st.markdown(f"- **RMSE (Root Mean Squared Error):** `{rmse:.2f}`")
                st.markdown(f"- **R² Score:** `{r2:.4f}`")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif selected == "Live Prediction":
    st.title("Live Next-Day Prediction")
    selected_stock = st.selectbox("Select a stock for live prediction", stockList)
    try:
        ticker = yf.Ticker(selected_stock)
        hist = ticker.history(period="20d")

        if len(hist) >= 10:
            close_prices = hist["Close"].dropna().values[-10:]
            close_prices = close_prices.reshape(-1, 1)

            scaled = scaler[selected_stock].transform(close_prices)
            model_input = np.array(scaled).reshape(1, 10, 1)

            next_scaled = model.predict(model_input)
            next_price = scaler[selected_stock].inverse_transform(next_scaled)[0][0]

            st.success(f"\U0001F4CC Predicted next closing price for **{selected_stock}**: **${next_price:.2f}**")

            dates = hist["Close"].dropna().index[-10:]
            next_day = dates[-1] + pd.Timedelta(days=1)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(dates, close_prices.flatten(), label="Last 10 Days", marker='o')
            ax2.plot(next_day, next_price, label="Predicted Next", marker='x', color='red')
            ax2.set_title("Recent Close Prices + Next Day Forecast")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price ($)")
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.warning("Not enough recent data available.")
    except Exception as e:
        st.error(f"Failed to fetch live data. Error: {e}")


elif selected == "News Sentiment":
    st.title("News Sentiment Analysis")
    selected_stock = st.selectbox("Select a stock for news sentiment", stockList)
    num_articles = st.slider("Number of news articles", 5, 20, 10)
    days_back = st.slider("Look back (days)", 1, 5, 1)

    def get_company_name(ticker):
        try:
            return yf.Ticker(ticker).info.get('longName', ticker)
        except:
            return ticker

    def fetch_news(query, num_results, days_back):
        googlenews = GoogleNews(lang='en')
        googlenews.set_time_range(f"{days_back}d", f"{days_back}d")
        googlenews.clear()
        googlenews.search(query)
        results = googlenews.result()[:num_results]
        headlines, links, dates = [], [], []
        for r in results:
            if r['title'].strip():
                headlines.append(r['title'])
                links.append(r.get('link', ''))
                try:
                    dates.append(pd.to_datetime(r.get('date', '')))
                except:
                    dates.append(pd.Timestamp.today())
        return headlines, links, dates

    if st.button("Analyze News Sentiment"):
        with st.spinner("Fetching and analyzing news..."):
            company_name = get_company_name(selected_stock)
            headlines, urls, dates = fetch_news(company_name, num_articles, days_back)

            if headlines:
                sentiments = predict_sentiment(headlines, return_probs=True)
                df = pd.DataFrame({
                    "Date": dates,
                    "Headline": headlines,
                    "Sentiment": [s[0] for s in sentiments],
                    "Confidence": [round(s[1], 3) for s in sentiments],
                    "URL": urls
                })

                st.dataframe(df)

                fig, ax = plt.subplots()
                df["Sentiment"].value_counts().plot.pie(
                    autopct="%1.1f%%", startangle=90, ax=ax,
                    colors=["red", "gray", "green"], explode=[0.05]*3
                )
                ax.set_ylabel("")
                ax.set_title(f"Sentiment Breakdown for {company_name}")
                st.pyplot(fig)
            else:
                st.warning("No news found for this stock.")
