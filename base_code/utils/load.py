import json
from git import Repo
from datasets import Dataset
from box import Box
import pandas as pd

def load_config(config_path: str):
    config_path = config_path + ".json"

    with open(config_path, 'r') as f:
        config = json.load(f)
        config = Box(config)

    return config

def load_last_commit(repo_path="."):
    try:
        repo = Repo(repo_path)
        return repo.head.commit.hexsha
    except Exception as e:
        print("Error while fetching the last commit hash:", e)
        return None