import pandas as pd
import numpy as np

def preprocess_data(path):

    # Impute missing values with mean and adjust class to 0&1
    data = pd.read_csv(path)
    data.drop(columns=['Time'], inplace=True)
    data.fillna(data.mean(), inplace=True)
    data['Pass/Fail'] = np.where(data['Pass/Fail'] == -1, 1, 0)
    return data

def load_data(path):
    data = preprocess_data(path)

    X = data.drop(columns=['Pass/Fail'])
    y = data['Pass/Fail']

    return X,y
