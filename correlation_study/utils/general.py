import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def save_results(results, output_path="results/statistics_results.yaml"):
    with open(output_path, "w") as file:
        yaml.dump(results, file)

