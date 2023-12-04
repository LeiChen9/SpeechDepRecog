import numpy as np 
import pandas as pd
import os
import pdb
from utils import *
import yaml

if __name__ == '__main__':
    # load data config
    with open("./configs/data_config.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    # load and parse data
    for key, value in data_config.items():
        with open(value['label'], 'r') as f:
            score = f.read()
            score = float(score)
            if score >= 53:
                label = 1
            else:
                label = 0
            pdb.set_trace()