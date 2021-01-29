def prepare_data(df, split, split_valid=None):

    df = split_train_test(df, split=split, split_valid=split_valid)

    df.rename(columns={"^BVSP": "IBOV"}, inplace=True)

    return df


def split_train_test(df, split, split_valid=None, ascending=True):

    # Number of rows
    nrows = df.shape[0]

    # Index to separate train and test datasets
    

    # Create label list
    if ascending == False:
        split_idx = round(split*nrows)
        label = ["train" if idx > split_idx else "test" for idx in range(nrows)]

        if split_valid is not None:
            split_idx_valid = round(split_valid*(nrows-split_idx)) + split_idx
            label = ["valid" if idx >= split_idx and idx <= split_idx_valid else \
                    "train" if idx > split_idx_valid \
                    else "test" \
                    for idx in range(nrows)]
    else:
        split_idx = round((1-split)*nrows)
        label = ["test" if idx > split_idx else "train" for idx in range(nrows)]

        if split_valid is not None:

            split_idx_valid = round(split_idx - round(split_valid*(nrows)))
            label = ["train" if idx < split_idx_valid else \
                    "valid" if idx >= split_idx_valid and idx <= split_idx \
                    else "test" \
                    for idx in range(nrows)]

    df["group"] = label

    return df