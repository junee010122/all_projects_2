import os
import sys
import yaml
import argparse

def parser_args(): 
    parser = argparse.ArgumentParser(description='Experiment Config')
    parser.add_argument('--configs', type=str, default='configs.yaml', help='/Users/june/Documents/code/all_projects_2/classification/configs')
    return parser

def load_params():

    parser = parser_args()
    args = parser.parse_args()
    from IPython import embed
    with open(args.configs, 'r') as f:
        params = yaml.safe_load(f)
    return params
