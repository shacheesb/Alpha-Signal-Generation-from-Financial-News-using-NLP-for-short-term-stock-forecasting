import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("📈 Financial News Sentiment Strategy Dashboard")

# --- Load real data exported from Jupyter ---
uploaded_file = st.sidebar.file_uploader("sentiment_strategy_data.csv", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data["Date"] = pd.to_datetime(data["Date"])

    # Ticker filter if included a ticker column
    if "ticker" in data.columns:
        tickers = data["ticker"].unique()
        selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)
        data = data[data["ticker"] == selected_ticker]

    # Optional: Date range filter
    date_range = st.sidebar.date_input("Select Date Range", [data["Date"].min(), data["Date"].max()])
    data = data[(data["Date"] >= pd.to_datetime(date_range[0])) & (data["Date"] <= pd.to_datetime(date_range[1]))]

    # Recalculate strategy return and cumulative growth if needed
    if "strategy_return" not in data.columns and "prediction" in data.columns:
        data["strategy_return"] = data["prediction"] * data["target_return"]

    data["cumulative_strategy"] = (1 + data["strategy_return"]).cumprod()
    data["cumulative_stock"] = (1 + data["target_return"]).cumprod()

    # Sharpe calculation
    def calc_sharpe(returns):
        returns = returns.dropna()
        if len(returns) < 2 or returns.std() == 0:
            return np.nan
        return (returns.mean() / returns.std()) * (252**0.5)

    sharpe_strategy = calc_sharpe(data["strategy_return"])
    sharpe_bh = calc_sharpe(data["target_return"])

    # Plot
    st.subheader("Cumulative Returns")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["Date"], data["cumulative_stock"], label="Buy & Hold")
    ax.plot(data["Date"], data["cumulative_strategy"], label="Model Strategy")
    ax.set_title("Cumulative Returns")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Sharpe Display
    col1, col2 = st.columns(2)
    col1.metric("Sharpe Ratio (Strategy)", f"{sharpe_strategy:.2f}" if not np.isnan(sharpe_strategy) else "N/A")
    col2.metric("Sharpe Ratio (Buy & Hold)", f"{sharpe_bh:.2f}" if not np.isnan(sharpe_bh) else "N/A")

    st.caption("Real data loaded from CSV. Make sure your file includes sentiment, features, and returns.")

else:
    st.info("👈 Upload your processed CSV from Jupyter to get started.")
