#  Alpha Signal Generation from Financial News

This project builds a pipeline to extract sentiment signals from financial news headlines and link them to stock price movements for potential predictive modeling. It combines NLP techniques, financial data analysis, and machine learning to explore the viability of using news-based sentiment as an alpha-generating signal.

## 📌 Project Overview

- **Goal**: Investigate whether sentiment derived from financial news headlines can predict short-term stock returns.
- **Methodology**:
  - Scrape or ingest historical financial news headlines
  - Apply sentiment analysis using VADER
  - Merge sentiment scores with stock price data from `yfinance`
  - Engineer features (lagged sentiment, moving averages, sector-based aggregation)
  - Build models to predict next-day or multi-day returns

## 🔧 Tech Stack

- **Python**
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- `yfinance` for stock price data
- `NLTK` / `VADER` for sentiment analysis
- Optional: `dbt`, `Snowflake`, `Looker`, or `Monte Carlo` for production-scale deployment

## 🧩 Key Features

- **Sentiment Scoring**: Assigns sentiment polarity to financial headlines using VADER.
- **Stock-Sentiment Merge**: Aligns timestamped news with corresponding tickers and price data.
- **Feature Engineering**: Lag features, sentiment volume, moving averages, and interaction terms.
- **Predictive Modeling**: Logistic regression, XGBoost, and Random Forests to classify next-day return direction.
- **Evaluation**: AUC, accuracy, precision, and lift over baseline models.

## 📈 Sample Workflow

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run sentiment processing
python sentiment_pipeline.py

# 3. Pull stock prices and merge
python merge_sentiment_with_prices.py

# 4. Train models
python train_model.py

# 5. View evaluation metrics
python evaluate.py
