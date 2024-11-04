import yaml
import os
from configs.default import get_config as get_default_config

def get_config(mode_string):
    config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f'{mode_string}_config.yml')
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    default_config = get_default_config()
    for k,v in config_dict.items():
        default_config[k].update(v)
    return default_config
