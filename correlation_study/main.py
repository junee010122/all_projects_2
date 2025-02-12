import os
import yaml
import pandas as pd
from tqdm import tqdm
from utils.data import load_data, preprocess_data
from utils.model import run_statistical_tests, train_model
from utils.plots import plot_correlation_heatmap, plot_boxplots, plot_roc_curve

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main(config_path):
    config = load_config(config_path)
    plot_results = config["experiment"].get("plot_results", False)
    
    df = load_data(config["experiment"]["dataset_path"])
    df = preprocess_data(df, config)
    results = run_statistical_tests(df, config)
   
    y = df.iloc[:, -1]  # Extract last column as target (Pass/Fail: 1 or -1)
    X = df.iloc[:, :-1]  # Extract all columns except the last one as features
    model = train_model(X, y, config)
    
    os.makedirs(config["experiment"]["output_path"], exist_ok=True)
    
    if plot_results:
        for step, func in tqdm(enumerate([plot_correlation_heatmap, plot_boxplots]), total=2):
            func(df, os.path.join(config["experiment"]["output_path"], f"plot_{step}.png"))
        
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X)[:, 1]
            plot_roc_curve(y, y_pred_proba, os.path.join(config["experiment"]["output_path"], "roc_curve.png"))

if __name__ == "__main__":
    main("configs/params.yaml")

