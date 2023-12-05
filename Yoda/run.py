import numpy as np 
import pandas as pd
import os
import pdb
import argparse
import torch
from dataloader import SDRDataLoader
from models.trans import DepressionClassifier
from utils import *
import yaml

if __name__ == '__main__':
    parser =  argparse.ArgumentParser(
        description="FunASR Common Training Parser",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--data_config", type=str, default="./configs/data_config.yaml", help="Data config file")
    parser.add_argument("--batch_size", type=int, default=8, help="Train batch size")
    parser.add_argument("--train_dtype", default="float32", choices=["float16", "float32", "float64"], help="Data type for training.")
    args = parser.parse_args()

    # set random seed
    set_all_random_seed(args.seed)

    # get data
    data_loader = SDRDataLoader(data_config_file=args.data_config, batch_size=args.batch_size)

    # get model
    model = SDRDataLoader()
    model = model.to(
        dtype=getattr(torch, args.train_dtype),
        device="cuda" if args.ngpu > 0 else "cpu",
    )
    