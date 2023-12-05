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

if __name__ == '__main__':
    parser =  argparse.ArgumentParser(
        description="FunASR Common Training Parser",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--data_config", type=str, default="./configs/data_config.yaml", help="Data config file")
    parser.add_argument("--batch_size", type=int, default=8, help="Train batch size")
    parser.add_argument("--train_dtype", default="float32", choices=["float16", "float32", "float64"], help="Data type for training.")
    args = parser.parse_args()

    # set writer
    writer = SummaryWriter('./log')

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
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # training params
    epochs = 50
    counter = 0
    # print_every = 500
    epoch_loss = 0
    for i in range(epochs):
        ########################
        #### Training part #####
        ########################
        epoch_loss = 0
        h = model.init_hidden(batch_size=args.batch_size)
        for inputs, labels in track(data_loader):
            counter += 1
            # in case last piece does not meet the batch size, so drop it.
            if(inputs.size(0) != args.batch_size):
                continue
            
            h = tuple([e.data for e in h])

            # inputs = inputs.double()
            # pdb.set_trace()

            model.zero_grad()

            output, h = model(inputs, h)

            loss = loss_function(output, labels.float())
            epoch_loss += loss.item()
            # pdb.set_trace()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()
        
        # Print train process info
        # mAP = eval(model)
        # writer.add_scalar('mAP', mAP, i)

        writer.add_scalar('epoch_train_loss', epoch_loss, global_step=i)

        ########################
        ###### Test part #######
        ########################
        # test_total_loss = 0
        # for inputs, labels in test_loader:
        #     if(inputs.size(0) != batch_size):
        #         continue
        #     h = tuple([each.data for each in h])
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     output, h = model(inputs, h)
        #     test_loss = loss_function(output, labels.float())
        #     test_total_loss += test_loss.item() 
        # # Print test process info 
        # writer.add_scalar('epoch_test_loss', test_total_loss, global_step=i)        

        # print("Epoch: {}/{}...".format(i + 1, epochs),
        #     "Step: {}...".format(counter),
        #     "Loss: {:.6f}...".format(loss.item()))
        
        writer.close()