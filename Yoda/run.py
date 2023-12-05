import numpy as np 
import pandas as pd
import os
import pdb
import argparse
import torch
import torch.nn as nn
from dataloader import SDRDataLoader
from models.trans import DepressionClassifier
from utils import *
from torch.utils.tensorboard import SummaryWriter 
from rich.progress import track
import yaml
from train_utils import make_train_step, make_validate_fnc, loss_fnc

if __name__ == '__main__':
    parser =  argparse.ArgumentParser(
        description="FunASR Common Training Parser",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--data_config", type=str, default="./configs/data_config.yaml", help="Data config file")
    parser.add_argument("--batch_size", type=int, default=8, help="Train batch size")
    parser.add_argument("--train_dtype", default="float32", choices=["float16", "float32", "float64"], help="Data type for training.")
    parser.add_argument("--ngpu", type=int, default=0)
    args = parser.parse_args()

    # set writer
    writer = SummaryWriter('./log')

    # set random seed
    set_all_random_seed(args.seed)

    # get data
    data_loader = SDRDataLoader(data_config_file=args.data_config, batch_size=args.batch_size)

    # training params
    EPOCHS=100
    DATASET_SIZE = len(data_loader.dataset)
    BATCH_SIZE = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device is {}'.format(device))

    # get model
    model = DepressionClassifier(num_targets=2).to(device)
    print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )
    OPTIMIZER = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

    train_step = make_train_step(model, loss_fnc, optimizer=OPTIMIZER)
    validate = make_validate_fnc(model,loss_fnc)
    losses=[]
    val_losses = []
    for epoch in range(EPOCHS):
        # schuffle data
        epoch_acc = 0
        epoch_loss = 0
        # iters = int(DATASET_SIZE / BATCH_SIZE)
        for batch in data_loader.data_loader:
            assert isinstance(batch, dict), type(batch)

            batch = batch.to(device)
            output_logits, output_softmax = model(**batch)
            