import os
from utils import *
import argparse
import librosa
import yaml 
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="mac", help="current machine env, mac or win or ubuntu")
    parser.add_argument("--config_file", type=str, default="./configs/data_config.yaml", help="data_config output")
    parser.add_argument("--scp_file", type=str, default="./configs/wav.scp", help='kaldi-style scp file')
    parser.add_argument("--overwrite", type=bool, default=False, help='config file overwrite')
    parser.add_argument("--source_dir", type=str, default="/Users/lei/Documents/Projs/Yoda/Data/EATD-Corpus/", help="path to source dir")
    args = parser.parse_args()
    # setup env 
    if args.env == 'ubuntu':
        args.config_file = './configs/ubuntu_data_config.yaml'
        args.scp_file = "./configs/ubuntu_wav.scp"
        args.source_dir = "/home/bix/Documents/Riceball/Code/yoda/Data/EATD-Corpus"
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
    
    # get rid of empty speech
    print("Getting rid of empty speech...")
    empty_dic = {}
    for uid, file_dic in data_config.items():
        for k, v in file_dic.items():
            if k == 'label':
                continue 
            curr_time = librosa.get_duration(filename=v)
            if curr_time == 0:
                empty_dic[uid] = k
    for uid, k in empty_dic.items():    
        print("{}-{}-{}".format(uid, k, data_config[uid].pop(k, None)))
    

    if not os.path.exists(args.config_file) or args.overwrite:
        with open(args.config_file, "w") as f:
            yaml.dump(data_config, f)

    with open(args.scp_file, "w") as f:
        for uid, file_dic in data_config.items():
            for k, v in file_dic.items():
                if k == 'label':
                    continue 
                curr_uid = '_'.join([uid, k])
                f.write(f"{curr_uid} {v}\n")
