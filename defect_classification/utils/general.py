import argparse
import yaml

def load_config():
    parser = argparse.ArgumentParser(description="MOSFET Defect Classification")
    parser.add_argument("--config", type=str, default="configs/configs.yaml", help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    print(f"Loaded config from {args.config}")
    return config

