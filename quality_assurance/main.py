import os
import torch   
import torch.nn as nn
import yaml
import pandas as pd
import sys
import time

from utils.general import load_config
from utils.data import preprocess_data
from utils.model import feature_selection, train_model

def run_experiments(params):    
    
    # Load the data
    data= pd.read_csv(params['paths']['data'])

    # perform data preprocessing
    X,y  = preprocess_data(data, params)
    
    # Load the model
    #run_stats(X, y)
    start_time = time.time()
    best_model, best_model_name = train_model(X, y, params)
    start_time = time.time()
    
    X_reduced = feature_selection(X, y, params)

    start_time = time.time()
    best_model_reduced, _ = train_model(X_reduced, y, params)
    reduced_train_time = time.time() - start_time

    # visualize results
    # plot_results(??)
    # streamlit and tableau bolognese

if __name__ == '__main__':
    params = load_config(sys.argv)
    #from IPython import embed; embed()
    run_experiments(params)
    
