import os
import torch   
import torch.nn as nn
import yaml
import pandas as pd
import sys


from utils.general import load_config
from utils.data import preprocess_data
#from utils.model import train_model, feature_selection

def run_experiemnts(params):    
    
    # Load the data
    data= pd.read_csv(params['paths']['data'])

    # perform data preprocessing
    exp_data = preprocess_data(data, params)
    from Iython import embed; embed()
    # Load the model
    model = train_model(sql_data, config)
    reduced_model = feature_selection(model, config)

    # visualize results
    # plot_results(??)
    # streamlit and tableau bolognese

if __name__ == '__main__':
    params = load_config(sys.argv)
    #from IPython import embed; embed()
    run_experiemnts(params)
    
