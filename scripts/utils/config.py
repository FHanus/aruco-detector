import os
import yaml

def load_config():
    """Load configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_script_config(script_name):
    """Get configuration for a specific script."""
    config = load_config()
    return config.get(script_name, {})