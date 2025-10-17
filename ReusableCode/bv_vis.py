## Required Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def biasVarianceCompFrame(df:pd.DataFrame, metrics:dict, 
                          cv_trials: int, parameters: dict,reverseMetricSign = False) -> pd.DataFrame:
    '''
    Reformats a GridSearchCV().cv_results_ Pandas DataFrame
    into another Pandas DataFrame by regrouping based on each fit.
    '''
    new_df = {key:[] for key in metrics|parameters} 
    new_df['Train/Test'] = []
    sgn = -2*reverseMetricSign + 1
    for row_ind in df.index:
        row = df.loc[row_ind]
        for i in range(cv_trials):
            for train_test in ['Train','Test']:
                for key,met in metrics.items():
                    col = 'split'+str(i)+'_'+train_test.lower()+'_'+met
                    new_df[key].append(sgn*row[col])
                new_df['Train/Test'].append(train_test)
                for key,param in parameters.items():
                    new_df[key].append(row[param])
    new_df = pd.DataFrame(new_df)
    return new_df

def plotBV(df:pd.DataFrame, metrics:list[str], title:str, 
           hues:list[str],complexity_param:str, subplotSize:tuple[float], style:str):
    """
    Visualizes the train-test errors for a model as a plot of the complexity parameter.
    """
    if hues:
        x,y = subplotSize
        n_rows, n_cols =len(hues), len(metrics)
        fig,ax = plt.subplots(ncols = n_cols, nrows = n_rows,figsize = (n_cols*x, n_rows*y),sharex = True)
        fig.suptitle(title)
        for row, hue in enumerate(hues):
            for col, met in enumerate(metrics):
                if col*row>0:
                    ind = [row,col]
                else:
                    ind = [row+col]
                sns.lineplot(data = df, y = met, hue = hue, x = complexity_param, ax = ax[*ind], style = style,palette = 'bright')
                if row==0:
                    ax[*ind].set_title(met)
                    ax[*ind].set_ylabel('')
    else:
        x,y = subplotSize
        n_rows, n_cols =1, len(metrics)
        fig,ax = plt.subplots(ncols = n_cols, nrows = n_rows,figsize = (n_cols*x, n_rows*y),sharex = True)
        fig.suptitle(title)
        row = 0
        for col, met in enumerate(metrics):
            if col*row>0:
                ind = [row,col]
            else:
                ind = [row+col]
            sns.lineplot(data = df, y = met, x = complexity_param, ax = ax[*ind], style = style,palette = 'bright')
            if row==0:
                ax[*ind].set_title(met)
                ax[*ind].set_ylabel('')
    plt.show()