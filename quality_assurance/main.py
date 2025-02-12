import os
import torch   
import torch.nn as nn
import yaml
import pandas as pd

from utils.general import load_config
from utils.data import convert_to_sql
from utils.model import train_model
from utils.visualization import plot_results

def run_experiemnts(params):    
    
    # Load the data
    data_path= pd.read_csv(params['data_path'])
    
    # convert data to sql
    sql_data = convert_to_sql(data)

    # perform data preprocessing
    preprocess_data(data, config)

    # Load the model
    model = train_model(sql_data, config)

    # visualize results
    # plot_results(??)
    # streamlit and tableau bolognese

if __name__ == '__main__':
    params = load_config(sys.argv)
    run_experiemnts(params)