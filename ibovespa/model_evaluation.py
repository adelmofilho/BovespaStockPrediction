import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_metrics(true, pred):
    
    # Predict delta accuracy
    delta_true = np.array([1 if np.sign(true[idx]-true[idx+1]) >=0 else 0 for idx in range(true.shape[0]-1)])
    delta_pred = np.array([1 if np.sign(pred[idx]-pred[idx+1]) >=0 else 0 for idx in range(pred.shape[0]-1)])
    
    # General metrics
    tp = sum(1 for val in (delta_true + delta_pred) if val==2)
    fp = sum(1 for val in (delta_true - delta_pred) if val==-1)
    fn = sum(1 for val in (delta_true - delta_pred) if val==1)
    precision = tp/(tp+fp)
    recal = tp/(tp+fn)
       
    # Model metrics
    mae = np.mean(np.abs(pred - true))
    f1 = 2*(precision*recal)/(precision+recal)
    
    return mae, f1


def model_prediction(model, x_tensor, y_tensor):

    pred = np.array(np.hstack(model(x_tensor).detach().numpy()).tolist())
    true = np.array(np.hstack(y_tensor).tolist())

    return true, pred


def benchmark_model(test_y_tensor, valid_y_tensor):

    true = np.array(list(np.array(np.hstack(test_y_tensor).tolist())))
    pred = np.array(list(np.array(valid_y_tensor[-1])) + list(np.array(np.hstack(test_y_tensor).tolist())[:-1]))

    return true, pred


def graphical_evaluation(model_true, model_pred, bmk_true, bmk_pred):

    modelo = pd.DataFrame(data={"true": model_true, "pred":model_pred})
    bmk = pd.DataFrame(data={"true": bmk_true, "pred":bmk_pred})

    f, axes = plt.subplots(3, 2, figsize=(16.5, 11.7))
    sns.despine(left=True)

    # Plot a simple distribution of the desired columns
    sns.scatterplot(x="index", y='true', data=modelo.reset_index(), ax=axes[0, 0])
    sns.lineplot(x="index", y="pred", data=modelo.reset_index(),ax=axes[0, 0], color='r')

    sns.scatterplot(x="index", y='true', data=bmk.reset_index(), ax=axes[0, 1])
    sns.lineplot(x="index", y="pred", data=bmk.reset_index(),ax=axes[0, 1], color='r')

    sns.lineplot(x="true", y="pred", data=modelo.reset_index(),ax=axes[1, 0], color='r')
    sns.lineplot(x="true", y="true", data=modelo.reset_index(),ax=axes[1, 0], color='blue')

    sns.lineplot(x="true", y="pred", data=bmk.reset_index(),ax=axes[1, 1], color='r')
    sns.lineplot(x="true", y="true", data=bmk.reset_index(),ax=axes[1, 1], color='blue')

    sns.histplot(data = modelo["true"] - modelo["pred"],ax=axes[2, 0], color='r')
    axes[2, 0].set_xlim(-1, 1)

    sns.histplot(data = bmk["true"] - bmk["pred"],ax=axes[2, 1], color='r')
    axes[2, 1].set_xlim(-1, 1)

    plt.setp(axes, yticks=[])
    plt.tight_layout()
