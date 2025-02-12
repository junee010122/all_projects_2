import os
import torch   
import torch.nn as nn
import yaml
import pandas as pd

from utils.data import convert_to_sql
from utils.model import train_model
from utils.visualization import plot_results

if __name__ == '__main__':
    # Load the configuration file
    with open('params.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    # Load the data
    data = pd.read_csv(config)
    
    # convert data to sql
    sql_data = convert_to_sql(data)

    # perform data preprocessing
    # preprocess_data(data, config)

    # Load the model
    model = train_model(sql_data, config)

    # visualize results
    # plot_results(??)
    # streamlit and tableau bolognese