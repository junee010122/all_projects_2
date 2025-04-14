import yaml
import os
import sys

from utils.data import load_data
from utils.model import train_model
from utils.general import load_config

def run_analysis(params):

    data_path = params['paths']['data']
    result_path = params['paths']['results']

    # Load data
    data = load_data(data_path)

    # Train model
    model = train_model(data, params)

    # Save model
    #model.save_model(params)

    # plot results
    #model.plot_results(data, params)
    return

if __name__ == "__main__":
    
    # Load config file
    params = load_config(sys.argv)
    run_analysis(params)
