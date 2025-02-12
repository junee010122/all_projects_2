# general.py - Utility functions
def load_config():
    import yaml
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)
