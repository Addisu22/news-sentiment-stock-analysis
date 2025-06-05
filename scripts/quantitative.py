import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import talib
import pandas as pd
# from pynance import Returns, Risk, Portfolio



def load_data_stock(file_path, ticker_name=None):
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

def apply_ta(df):

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


def fin_met(df):
  
  # Prepare data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Calculate daily log returns
    df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    # Annualized volatility (std dev of returns * sqrt trading days)
    volatility = df['Log_Returns'].std() * np.sqrt(252)

    # Annualized Sharpe Ratio assuming risk-free rate = 0
    sharpe_ratio = (df['Log_Returns'].mean() / df['Log_Returns'].std()) * np.sqrt(252)

    # Calculate max drawdown
    cumulative = (1 + df['Log_Returns']).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    met = {
        "Annualized Volatility": volatility,
        "Annualized Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown
    }
    
    return met




def visualize_indicators(df, ticker_name="Stock"):
  
    # Convert 'Date' to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Calculate indicators
    df['SMA20'] = talib.SMA(df['Adj Close'], timeperiod=20)
    df['SMA50'] = talib.SMA(df['Adj Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal

    # Plot Closing Price with Moving Averages
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Adj Close'], label='Adj Close', color='black')
    plt.plot(df['Date'], df['SMA20'], label='SMA 20', color='blue', linestyle='--')
    plt.plot(df['Date'], df['SMA50'], label='SMA 50', color='green', linestyle='--')
    plt.title(f'{ticker_name} - Adj Close Price & Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Adj Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot RSI
    plt.figure(figsize=(14, 3))
    plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'{ticker_name} - Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.grid(True)
    plt.show()

    # Plot MACD
    plt.figure(figsize=(14, 4))
    plt.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    plt.plot(df['Date'], df['MACD_Signal'], label='Signal', color='red')
    plt.title(f'{ticker_name} - MACD')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()
    plt.grid(True)
    plt.show()

