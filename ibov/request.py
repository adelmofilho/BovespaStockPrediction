from yahooquery import Ticker


def get_history(ticker, data_size, ascending):
     
     ticker = Ticker(symbols = ticker)
     history = ticker.history(period="max")
     
     df = history.sort_values(by="date", ascending=ascending)
          
     if ascending:
          df = df.tail(data_size).reset_index(drop=False)
     else:
          df = df.head(data_size).reset_index(drop=False)
          
     return df

