import pandas as pd
import numpy as  np   
import textblob as tb  
import matplotlib.pyplot as plt
import seaborn as sns  
import logging


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


def headline_length_stats(df, headline_col='Headline'):
    try:
        if headline_col not in df.columns:
            raise ValueError(f"Column '{headline_col}' not found in dataframe.")
        
        # Compute length of each headline string
        df['headline_length'] = df[headline_col].astype(str).apply(len)
        
        # Return basic descriptive statistics
        return df['headline_length'].describe()
    except Exception as e:
        print(f"Error in headline_length_stats: {e}")
        return None
    

def count_articles_per_publisher(df, publisher_col='Publisher'):
    try:
        if publisher_col not in df.columns:
            raise ValueError(f"Column '{publisher_col}' not found in dataframe.")
        
        return df[publisher_col].value_counts()
    except Exception as e:
        print(f"Error in count_articles_per_publisher: {e}")
        return None

