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
        ind = np.random.permutation(DATASET_SIZE)
        X_train = X_train[ind,:,:,:]
        Y_train = Y_train[ind]
        epoch_acc = 0
        epoch_loss = 0
        iters = int(DATASET_SIZE / BATCH_SIZE)
        for i in range(iters):
            batch_start = i * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
            actual_batch_size = batch_end-batch_start
            X = X_train[batch_start:batch_end,:,:,:]
            Y = Y_train[batch_start:batch_end]
            X_tensor = torch.tensor(X,device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
            loss, acc = train_step(X_tensor,Y_tensor)
            epoch_acc += acc*actual_batch_size/DATASET_SIZE
            epoch_loss += loss*actual_batch_size/DATASET_SIZE
            print(f"\r Epoch {epoch}: iteration {i}/{iters}",end='')
        X_val_tensor = torch.tensor(X_val,device=device).float()
        Y_val_tensor = torch.tensor(Y_val,dtype=torch.long,device=device)
        val_loss, val_acc, predictions = validate(X_val_tensor,Y_val_tensor)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        print('')
        print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.}")