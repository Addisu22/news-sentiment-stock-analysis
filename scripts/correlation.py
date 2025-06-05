import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import talib
import pandas as pd
import textblob as tb 
from scipy.stats import pearsonr


# Function to load and preprocess news data
def load_news_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # Ensure published_date exists
        if 'date' not in df.columns:
            print("'date' column not found.")
            return None
        
        # Convert to datetime and normalize to date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['Date'] = df['date'].dt.date
        
        return df
    except Exception as e:
        print(f"Error loading news data: {e}")
        return None
    

# Function to load and preprocess stock data
def load_stock_data(file_path, ticker_name=None):
    try:
        df = pd.read_csv(file_path)
        
        if 'Date' not in df.columns:
            print("'Date' column not found in stock data.")
            return None
        
        # Normalize stock dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        df = df.dropna(subset=['Date'])

          # Add Ticker name column if provided
        if ticker_name:
            df['Ticker'] = ticker_name
        
        return df
    except Exception as e:
        print(f"Error loading stock data {file_path}: {e}")
        return None


def align_by_date(news_df, stock_df):
    try:
        if news_df is None or stock_df is None:
            print("One of the datasets is missing.")
            return None, None

        # Merge on normalized date
        aligned_df = pd.merge(news_df, stock_df, on='Date', how='inner')
        return aligned_df
    except Exception as e:
        print(f"Error aligning datasets: {e}")
        return None
    


# Load your news data (replace path as needed)
def load_news(filepath):
    try:
        df = pd.read_csv(filepath)
        if 'headline' not in df.columns:
            raise ValueError("Missing 'headline' column.")
        df.dropna(subset=['headline'], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading news data: {e}")
        return pd.DataFrame()

# Function to calculate sentiment polarity
def get_sentiment_polarity(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

# Function to classify polarity score
def classify_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
def sentiment_analysis(news_df):
  try:
        news_df['sentiment_score'] = news_df['headline'].apply(get_sentiment_polarity)
        news_df['sentiment_label'] = news_df['sentiment_score'].apply(classify_sentiment)
        return news_df
  except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return news_df
  

  import pandas as pd

# def calculate_daily_returns(df, price_column='Adj Close', date_column='Date'):
#     try:
#         if price_column not in df.columns:
#             raise KeyError(f"'{price_column}' column not found in DataFrame.")
#         if date_column not in df.columns:
#             raise KeyError(f"'{date_column}' column not found in DataFrame.")

#         # Ensure date is datetime and sort
#         df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
#         df = df.dropna(subset=[date_column, price_column])
#         df = df.sort_values(by=date_column)

#         # Calculate returns
#         df['daily_return'] = df[price_column].pct_change()

#         return df
#     except Exception as e:
#         print(f"Error calculating daily returns: {e}")
#         return pd.DataFrame()


# Function to calculate daily returns
def calculate_daily_returns(df, price_column='Adj Close', date_column='Date'):
    try:
        if price_column not in df.columns:
            raise KeyError(f"'{price_column}' column not found in DataFrame.")
        if date_column not in df.columns:
            raise KeyError(f"'{date_column}' column not found in DataFrame.")

        # Ensure date is datetime and sort
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column, price_column])
        df = df.sort_values(by=date_column)

        # Calculate returns
        df['Daily_return'] = df[price_column].pct_change()

        # Keep only relevant columns
        return df[[date_column, 'Daily_return']].rename(columns={'Daily_return': f"{ticker}_return"})

    except Exception as e:
        print(f"Error calculating daily returns for {ticker}: {e}")
        return pd.DataFrame()

# List of stock tickers and their file paths
tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA']
file_path_template = "Data/{}_historical_data.csv"

# DataFrame to store aligned returns
returns_df = pd.DataFrame()

for ticker in tickers:
    try:
        df = pd.read_csv(file_path_template.format(ticker))
        return_df = calculate_daily_returns(df)
        if not return_df.empty:
            if returns_df.empty:
                returns_df = return_df.set_index('Date')
            else:
                return_df = return_df.set_index('Date')
                returns_df = returns_df.join(return_df, how='outer')  # outer join ensures all dates
    except FileNotFoundError:
        print(f"⚠️ File not found for {ticker}: {file_path_template.format(ticker)}")

# Reset index for inspection
returns_df.reset_index(inplace=True)

# Preview
print("Combined daily returns for 7 stocks:")
print(returns_df.head())



def correlate_sentiment_returns(sentiment_df, stock_df,
                                date_column='date',
                                sentiment_column='sentiment_score',
                                return_column='daily_return'):
    try:
        # Check for necessary columns
        for col in [date_column, sentiment_column]:
            if col not in sentiment_df.columns:
                raise KeyError(f"'{col}' not found in sentiment DataFrame.")
        for col in [date_column, return_column]:
            if col not in stock_df.columns:
                raise KeyError(f"'{col}' not found in stock DataFrame.")

        # Convert dates to datetime for alignment
        sentiment_df[date_column] = pd.to_datetime(sentiment_df[date_column])
        stock_df[date_column] = pd.to_datetime(stock_df[date_column])

        # Merge datasets on date
        merged_df = pd.merge(sentiment_df[[date_column, sentiment_column]],
                             stock_df[[date_column, return_column]],
                             on=date_column, how='inner').dropna()

        # Calculate correlation
        correlation, p_value = pearsonr(merged_df[sentiment_column],
                                        merged_df[return_column])

        return {
            'correlation_coefficient': correlation,
            'p_value': p_value,
            'merged_sample_size': len(merged_df)
        }

    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        return None
