import pandas as pd


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