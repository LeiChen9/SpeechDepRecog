import os
from utils import *
import argparse
import yaml 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./configs/data_config.yaml", help="data_config output")
    parser.add_argument("--overwrite", type=bool, default=False, help='config file overwrite')
    parser.add_argument("--source_dir", type=str, default="/Users/lei/Documents/Projs/Yoda/Data/EATD-Corpus/", help="path to source dir")
    args = parser.parse_args()
    # load data
    data_dir = args.source_dir
    data_config = {}
    for subdir in os.listdir(data_dir):
        if check_string('[a-z]_[0-9]+', subdir):
            curr_dir = os.path.join(data_dir, subdir)
            pos_file_name = os.path.join(curr_dir, 'positive_out.wav')
            neg_file_name = os.path.join(curr_dir, 'negative_out.wav')
            neutral_file_name = os.path.join(curr_dir, 'neutral_out.wav')
            label_file_name = os.path.join(curr_dir, 'label.txt')
            data_config[subdir] = {
                'pos': pos_file_name,
                'neg': neg_file_name,
                'neutral': neutral_file_name,
                'label': label_file_name
            }
    if not os.path.exists(args.config_file) or args.overwrite:
        with open(args.config_file, "w") as f:
            yaml.dump(data_config, f)
