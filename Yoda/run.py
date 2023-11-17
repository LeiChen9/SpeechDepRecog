import os 
import torch
import utils
import torch.multiprocessing as mp
from data_utils import TextAudioCollate, TextAudioLoader

def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))

def run(rank, n_gpus, hps):
    hps = utils.get_hparams()
    collate_fn = TextAudioCollate()
    train_dataset = TextAudioLoader(hps.train_folder, hps)

if __name__ == '__main__':
    main()