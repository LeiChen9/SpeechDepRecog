import os 
import torch
import utils
import pdb
import torch.distributed as dist
import torch.multiprocessing as mp
from data_utils import TextAudioCollate, TextAudioLoader
from torch.utils.tensorboard import SummaryWriter

def main():
    """Assume Single Node Multi GPUs Training Only"""
    # assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port
    # pdb.set_trace()
    # mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
    run(1, 1, hps)

def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # for pytorch on win, backend use gloo    
    # dist.init_process_group(backend=  'gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    # torch.manual_seed(hps.train.seed)
    # torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem
    # pdb.set_trace()
    train_dataset = TextAudioLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)

if __name__ == '__main__':
    main()