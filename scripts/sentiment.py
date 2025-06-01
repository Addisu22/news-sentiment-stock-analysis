import pandas as pd
import numpy as  np   
import textblob as tb  
import matplotlib.pyplot as plt
import seaborn as sns  


def load_data(filepath):
    return pd.read_csv(filepath)

def compute_sentiment(df, text_col="headline"):
    df = df.copy()
    df["Polarity"] = df[text_col].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    df["Subjectivity"] = df[text_col].astype(str).apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

def filter_company_news(df, companies):
    result = {}
    for company in companies:
        filtered = df[df["headline"].str.contains(company, case=False, na=False)]
        result[company] = filtered
    return result

def summarize_sentiment(df):
    return df["Polarity"].describe()