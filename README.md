#  Alpha Signal Generation from Financial News

This project builds a pipeline to extract sentiment signals from financial news headlines and link them to stock price movements for potential predictive modeling. It combines NLP techniques, financial data analysis, and machine learning to explore the viability of using news-based sentiment as an alpha-generating signal and deploys it with the help of Streamlit app.

## 📌 Project Overview

- **Goal**: Investigate whether sentiment derived from financial news headlines can predict short-term stock returns.
- **Methodology**:
  - Scrape or ingest historical financial news headlines
  - Apply sentiment analysis using VADER
  - Merge sentiment scores with stock price data from `yfinance`
  - Engineer features (lagged sentiment, moving averages, sector-based aggregation)
  - Build models to predict next-day or multi-day returns
  - Display results/forecast on streamlit dashboard

## 🔧 Tech Stack

- **Python**
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- `yfinance` for stock price data
- `NLTK` / `VADER` for sentiment analysis
- Streamlit for UI dashboard

## 🧩 Key Features

- **Sentiment Scoring**: Assigns sentiment polarity to financial headlines using VADER.
- **Stock-Sentiment Merge**: Aligns timestamped news with corresponding tickers and price data.
- **Feature Engineering**: Lag features, sentiment volume, moving averages, and interaction terms.
- **Predictive Modeling**: Logistic regression, XGBoost, and Random Forests to classify next-day return direction.
- **Evaluation**: AUC, accuracy, precision, and lift over baseline models.
- **Deployment** Streamlit UI
