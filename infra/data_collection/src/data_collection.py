import pandas as pd
from yahooquery import Ticker


def get_history(ticker, data_size, ascending=True):
     
     ticker = Ticker(symbols = ticker)
     history = ticker.history(period="max")
     
     df = history.sort_values(by="date", ascending=ascending)
     
     if ascending:
          df = df.tail(data_size).reset_index(drop=False)
     else:
          df = df.head(data_size).reset_index(drop=False)
     
     return df[["date", "close"]]  


def collect_data(stocks, data_size, ascending=True):
    
     stocks_df = get_history(stocks[0], data_size, ascending)
     stocks_df.rename(columns={"close": stocks[0]}, inplace=True)
    
     for stock in stocks[1:]:
         stock_df = get_history(stock, data_size, ascending)
         stock_df.rename(columns={"close": stock[:-3]}, inplace=True)
         stocks_df = stocks_df.merge(stock_df, how="inner", on="date")
        
     return stocks_df