import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_target(dados, target):

    scaler = MinMaxScaler()
    dados[[target]] = scaler.fit_transform(dados[[target]])

    return dados


def label_train_test(df, split, split_valid=None, ascending=True):

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

                   # and idx <= split_idx else \
    # Add train/test label on dataset
    df["group"] = label

    return df


def create_lags(data, window, var, index):
    
    df = pd.DataFrame(data[index][:-window], columns = [index])

    y_list = list()
    x_list = list()

    for idx in range(data.shape[0]-window):
        y = [data[var][idx]]
        x = list(data[var][(idx+1):(window+idx+1)].values)
        y_list.append(y)
        x_list.append(x)

    df['target'] = y_list
    df['lags'] = x_list
    
    return df


def consolidate_features(base, index, *args):
    
    for feature in args:
        base = base.merge(feature, how='inner', on=index)
        
    return base


def create_delta_sign(data, var, index, window):
    
    data = data.copy()
    
    delta_sign = \
        np.reshape([np.sign(i-j) for t in data[var] for i, j in zip(t[:-1], t[1:])], (data.shape[0], window-1))
    
    result = [a+[x] for a,x in zip([l.tolist() for l in delta_sign], [0]*delta_sign.shape[0])]
    
    data["delta_sign"] = result
    
    delta_sign_df = data[[index, "delta_sign"]]  
    
    return delta_sign_df