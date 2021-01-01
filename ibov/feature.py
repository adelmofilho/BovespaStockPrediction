def create_lags(data, window):

    y_list = list()
    x_list = list()

    for idx in range(data.shape[0]-window):
        y = [data[idx]]
        x = list(data[(idx+1):(window+idx+1)].values)
        y_list.append(y)
        x_list.append(x)
    
    return y_list, x_list
