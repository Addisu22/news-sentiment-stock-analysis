import pandas as pd
import numpy as  np   
import textblob as tb  
import matplotlib.pyplot as plt
import seaborn as sns  
import logging
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


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


def headline_length_stats(df, headline='headline'):
    try:
        if headline not in df.columns:
            raise ValueError(f"Column '{headline}' not found in dataframe.")
        
        # Compute length of each headline string
        df['headline_length'] = df[headline].astype(str).apply(len)
        
        # Return basic descriptive statistics
        return df['headline_length'].describe()
    except Exception as e:
        print(f"Error in headline_length_stats: {e}")
        return None
    

def count_articles_per_publisher(df, publisher='publisher'):
    try:
        if publisher not in df.columns:
            raise ValueError(f"Column '{publisher}' not found in dataframe.")
        
        return df[publisher].value_counts()
    except Exception as e:
        print(f"Error in count_articles_per_publisher: {e}")
        return None


def publication_trend(df, date='date', freq='D'):
    try:
        if date not in df.columns:
            raise ValueError(f"Column '{date}' not found in dataframe.")
        
        # Convert to datetime
        df[date] = pd.to_datetime(df[date], errors='coerce')
        df = df.dropna(subset=[date])
        
        # Group and count articles by time period
        counts = df.groupby(df[date].dt.to_period(freq)).size()
        counts.index = counts.index.to_timestamp()
        counts.name = 'article_count'
        return counts
    except Exception as e:
        print(f"Error in publication_trend: {e}")
        return None
    

def preprocess_text(df, text_col='headline'):  
      try:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in dataframe.")
        
        df = df.copy()
        df[text_col] = df[text_col].astype(str).str.lower()
        df[text_col] = df[text_col].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
        return df
      except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return None
      

def extract_topics(df, text_col='headline', num_topics=5, num_words=10):
      try:
        if df is None or text_col not in df.columns:
            raise ValueError("Valid DataFrame with text column is required.")

        # Vectorize the text
        vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
        doc_term_matrix = vectorizer.fit_transform(df[text_col])
        
        # Apply LDA
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(doc_term_matrix)

        # Extract topics
        words = vectorizer.get_feature_names_out()
        topics = []
        for i, topic in enumerate(lda.components_):
            topic_keywords = [words[i] for i in topic.argsort()[:-num_words - 1:-1]]
            topics.append(f"Topic {i+1}: " + ", ".join(topic_keywords))
        return topics
      except Exception as e:
        print(f"Error in extract_topics: {e}")
        return None



def analyze_publication_frequency(df):
    try:
        df['date'] = pd.to_datetime(df['date'])
        daily_counts = df['date'].dt.date.value_counts().sort_index()
        daily_counts.plot(figsize=(12, 5), title='Article Count Over Time', xlabel='Date', ylabel='Articles')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return daily_counts
    except Exception as e:
        print(f"Error in publication frequency analysis: {e}")
        return None

def analyze_publishing_time(df):
    try:
        if 'date' not in df.columns:
            print("Warning: 'time' column not found. Publishing time analysis skipped.")
            return None
        
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df['hour'] = df['datetime'].dt.hour
        hourly_counts = df['hour'].value_counts().sort_index()
        
        hourly_counts.plot(kind='bar', figsize=(10, 5), title='Article Count by Hour of Day', xlabel='Hour', ylabel='Articles')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        return hourly_counts
    except Exception as e:
        print(f"Error in publishing time analysis: {e}")
        return None
    

def top_publishers(df, top_n=10):
    try:
        publisher_counts = df['publisher'].value_counts().head(top_n)
        return publisher_counts
    except Exception as e:
        print(f"Error extracting top publishers: {e}")
        return None

def extract_email_domains(df):
    try:
        email_publishers = df['publisher'].dropna()
        domain_pattern = r'@([\w\.-]+\.\w+)'
        domains = email_publishers[email_publishers.str.contains('@')].apply(
            lambda x: re.findall(domain_pattern, x.lower())
        )
        domain_list = [d[0] for d in domains if d]
        domain_counts = pd.Series(Counter(domain_list)).sort_values(ascending=False)
        return domain_counts
    except Exception as e:
        print(f"Error extracting email domains: {e}")
        return None
    

def day_of_week_analysis(df):
    try:
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['day_of_week'] = df['published_date'].dt.day_name()
        return df['day_of_week'].value_counts()
    except Exception as e:
        print(f"Error in weekday analysis: {e}")
        return None

def daily_article_counts(df):
    try:
        df['published_date'] = pd.to_datetime(df['published_date'])
        counts = df['published_date'].dt.date.value_counts().sort_index()
        return counts
    except Exception as e:
        print(f"Error in daily article analysis: {e}")
        return None