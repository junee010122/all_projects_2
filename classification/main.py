import os
import sys
import pickle as pkl
import numpy as np
import yaml

from utils.general import load_params
from utils.data import load_data
from utils.models import train_model
from utils import models

def run_experiment(params):

    # Load data
    data_path = params['paths']['data']
    X,y = load_data(data_path)

    # Load model
    model = train_model(X, y, params)
if __name__ == "__main__":
    
    params = load_params() 
    run_experiment(params)
