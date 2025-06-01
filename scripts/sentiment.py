import pandas as pd
import numpy as  np   
import textblob as tb  
import matplotlib.pyplot as plt
import seaborn as sns  
import logging


def load_news_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        return df
    except Exception as e:
        print(f"[Error loading data] {e}")
        return None

def headline_length_stats(df):
    try:
        df['Length'] = df['Headline'].astype(str).apply(len)
        return df['Length'].describe()
    except Exception as e:
        print(f"[Error in headline length stats] {e}")
        return None

def article_count_per_publisher(df):
    try:
        return df['Publisher'].value_counts()
    except Exception as e:
        print(f"[Error counting articles per publisher] {e}")
        return None

def publication_trend(df, freq='D'):
    # try:
    #     return df.groupby(df['Date'].dt.to_period(freq)).size().rename("Article_Count")
    # except Exception as e:
    #     print(f"[Error analyzing publication trend] {e}")
    #     return None
    try:
        if 'Date' not in df.columns:
            raise ValueError("Missing 'Date' column.")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        grouped = df.groupby(df['Date'].dt.to_period(freq)).size().rename("Article_Count")
        return grouped
    except Exception as e:
        print(f"[Error analyzing publication trend] {e}")
        return pd.Series()





def load_data(data_path):

    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded data with shape: {df.shape}")

        # Ensure required columns exist
        required_cols = ['headline', 'date']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: '{col}'")

        # Convert date to datetime, coerce errors
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Drop rows with missing required data
        df = df.dropna(subset=['headline', 'date'])
        logging.info(f"Data after dropping missing headline/date rows: {df.shape}")

        return df

    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error("File is empty")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def compute_sentiment(df, text_col="headline"):
    df = df.copy()
    # Add sentiment polarity and subjectivity
    df["Polarity"] = df[text_col].astype(str).apply(lambda x: tb(x).sentiment.polarity)
    df["Subjectivity"] = df[text_col].astype(str).apply(lambda x: tb(x).sentiment.subjectivity)
    return df

def filter_company_news(df, companies):
    result = {}
    for company in companies:
        filtered = df[df["headline"].str.contains(company, case=False, na=False)]
        result[company] = filtered
    return result

def summarize_sentiment(df):
    return df["Polarity"].describe()