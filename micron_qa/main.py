import os
import sys
import pickle as pkl
import numpy as np
import yaml

from utils.general import load_params
from utils.data import load_data
from utils import tab_model, img_model

def run_experiment(params): 
    # Load data
    data = load_data(params)
    # Train and evaluate model based on datatype
    if params['system']['datatype'] == 1:  # image
        model_type = params['model']['image'].get('model_type', 0)
        model = img_model.ResNet18(num_classes=params['task']['image']['num_classes'])  # for now, assume ResNet18
        model = img_model.train_model(model, data, params)
        img_model.evaluate_model(model, data, params)

    else:  # tabular
        model = tab_model.train_model(data, params)
        tab_model.evaluate_model(model, data, params)

    # Optional: Plotting (not enabled here)
    # if params['plots']['results']:
    #     plot_results(model, data, params)

    return None

if __name__ == '__main__':
    params = load_params()
    run_experiment(params)

