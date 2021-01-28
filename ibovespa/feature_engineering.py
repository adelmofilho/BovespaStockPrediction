import pandas as pd
import numpy as np

def engineer_features(dados, window, mode, target, model=None):

    # Target Normalization

    scaler = Normalize()

    if mode == "train":
        scaler.fit(dados[dados["group"]=="train"][[target]])
    elif mode == "predict":
        scaler.load_configs(maximo=model.maximo, minimo=model.minimo)
    else:
        raise Exception("mode does not exist")

    dados[[target]] = scaler.transform(dados[[target]])

    # Feature Engineering   

    ibov_lags_df = create_lags(dados, 
                               window=window, 
                               var=target, 
                               index="date")

    ibov_delta_sign_df = create_delta_sign(ibov_lags_df, 
                                           var="lags", 
                                           index="date", 
                                           window=window)
    
    weekdays_df = create_weekday(dados, window)

    day_change_df = create_day_change(dados, window)
    
    master_table = consolidate_features(dados[["date", "group"]], "date", 
                                        ibov_lags_df, 
                                        ibov_delta_sign_df,
                                        weekdays_df,
                                        day_change_df)    

    return master_table, scaler 


def create_weekday(df, window):
    dates = df[["date"]]
    df = pd.get_dummies(pd.DatetimeIndex(df['date']).weekday, prefix="weekday")
    diff_window = window - df.shape[1]
    
    if diff_window > 0:
        for idx in range(diff_window):
            df["dummy_" + str(idx)] = 0
            
    df['weekday_vector']= df.values.tolist()
    df = pd.concat([dates, df], axis=1)
    
    return df[["date", "weekday_vector"]]


class Normalize():

    def __init__(self, type="minmax"):

        self.type = type

    def fit(self, data):

        self.data = data
        self.maximo = data.values.max()
        self.minimo = data.values.min()
    
    def transform(self, data):

        self.transformed = (data - self.minimo)/(self.maximo - self.minimo)

        return self.transformed

    def denormalize(self, data):

        self.detransformed = data*(self.maximo - self.minimo) + self.minimo

        return self.detransformed

    def load_configs(self, maximo, minimo):

        self.maximo = maximo
        self.minimo = minimo


def create_day_change(df, window):
    
    variables = list(df.columns)[1:-1]
    stocks_diff = df[variables].pct_change().reset_index(drop=True)
    diff_dates = df.iloc[2:][["date"]].reset_index(drop=True)
    last_diff_df = stocks_diff.iloc[1:].iloc[:-1].reset_index(drop=True)
    last_day_stock_diff = pd.concat([diff_dates, last_diff_df], axis=1)
    
    
    data = last_day_stock_diff.sort_values(by="date", ascending=False).reset_index(drop=True)
    df = pd.DataFrame(data[index][:-window], columns = [index])

    for var in variables:    
        x_list = list()
        for idx in range(data.shape[0]-window):
            x = list(data[var][(idx):(window+idx)].values)
            x_list.append(x)

        df['lag_pct_'+var] = x_list
    
    return df.sort_values(by="date", ascending=True).reset_index(drop=True)



def create_lags(data, window, var, index):
    
    data = data.sort_values(by="date", ascending=False).reset_index(drop=True)
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
    
    return df.sort_values(by="date", ascending=True).reset_index(drop=True)


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