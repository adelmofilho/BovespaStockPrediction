from yahooquery import Ticker


def get_history(ticker):

    ticker = Ticker(symbols = ticker)
    history = ticker.history(period="max")
    
    df = history.\
         sort_values(by="date", ascending=False).\
         reset_index(drop=False)

    return df


def label_train_test(df, split, split_valid=None):

    # Number of rows
    nrows = df.shape[0]

    # Index to separate train and test datasets
    split_idx = round(split*nrows)

    # Create label list
    label = ["train" if idx > split_idx else "test" for idx in range(nrows)]

    if split_valid is not None:
        split_idx_valid = round(split_valid*(nrows-split_idx)) + split_idx
        label = ["valid" if idx >= split_idx and idx <= split_idx_valid else \
                "train" if idx > split_idx_valid \
                else "test" \
                for idx in range(nrows)]

    # Add train/test label on dataset
    df["group"] = label

    return df

