import json
from datasets import Dataset
from box import Box
import pandas as pd

def load_config(config_path: str):
    config_path = config_path + ".json"

    with open(config_path, 'r') as f:
        config = json.load(f)
        config = Box(config)

    return config