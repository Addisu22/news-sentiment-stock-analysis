import os
import pandas as pd
import matplotlib.pyplot as plt
import talib

def load_data(file_paths):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        ticker = os.path.basename(path).split(".")[0].upper()
        df = df[['Date', 'Adj Close']].copy()
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        dfs.append(df)
    
    # Merge on Date
    combined_df = dfs[0]
    for other_df in dfs[1:]:
        combined_df = pd.merge(combined_df, other_df, on='Date', how='outer')
    
    combined_df.sort_values('Date', inplace=True)
    combined_df.set_index('Date', inplace=True)
    return combined_df

def visualize_data(file_paths):
    # Combine all CSVs
    df_list = []
    for file in file_paths:
        temp_df = pd.read_csv(file)
        temp_df['Ticker'] = file.split("/")[-1].split(".")[0]  # Extract ticker from filename
        df_list.append(temp_df)

    df = pd.concat(df_list)
    df['Date'] = pd.to_datetime(df['Date'])

    # Visualization
    pivot_df = df.pivot(index='Date', columns='Ticker', values='Adj Close')
    pivot_df.plot(figsize=(12, 6), title="Adjusted Close Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adj Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df[['Date', 'Adj Close', 'Ticker']]

def apply_talib_indicators(df):

    # Check if necessary columns exist
    if 'Adj Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Adj Close' column.")
    
    # Simple Moving Average (SMA) over 20 days
    df['SMA_20'] = talib.SMA(df['Adj Close'], timeperiod=20)
    
    # Relative Strength Index (RSI) with 14-day window
    df['RSI_14'] = talib.RSI(df['Adj Close'], timeperiod=14)
    
    # MACD calculation
    macd, macdsignal, macdhist = talib.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist
    
    return df