import pandas as pd
import numpy as  np   
import textblob as tb    


# Load the dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Compute sentiment scores
def compute_sentiment(df, text_col="headline"):
    df[text_col] = df[text_col].astype(str)
    df["Polarity"] = df[text_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["Subjectivity"] = df[text_col].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

# Filter news for a specific company
def filter_company_news(df, company_name):
    return df[df["headline"].str.contains(company_name, case=False, na=False)]

# Summarize sentiment
def summarize_sentiment(df):
    return {
        "average_polarity": df["Polarity"].mean(),
        "average_subjectivity": df["Subjectivity"].mean(),
        "most_positive": df.loc[df["Polarity"].idxmax()]["headline"],
        "most_negative": df.loc[df["Polarity"].idxmin()]["headline"],
    }