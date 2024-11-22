import argparse
from pipeline import BasePipeline
from manager import GemmaManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/baseline", help="path where config json is store")
    args = parser.parse_args()

    pipeline = BasePipeline(args.config, GemmaManager)
    pipeline.train()