from yahooquery import Ticker


def get_history(ticker):

    ticker = Ticker(symbols = ticker)
    history = ticker.history(period="max")
    
    df = history.\
         sort_values(by="date", ascending=False).\
         reset_index(drop=False)

    return df

