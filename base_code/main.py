import torch
import random
import numpy as np
import argparse
from utils.load import load_config

from trainer import MyTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config", help="path where config json is store")
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["settings"]
    trainer = MyTrainer(settings["dataset"], settings["model_name"], config["params"])
    trainer.train()