import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib
import pandas as pd
import textblob as tb 


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
        aligned_df = pd.merge(news_df, stock_df, on='date', how='inner')
        return aligned_df
    except Exception as e:
        print(f"Error aligning datasets: {e}")
        return None
