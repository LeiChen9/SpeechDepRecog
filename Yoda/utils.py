import re 
import random 
import torch
import numpy as np

def check_string(re_exp, str):
    res = re.search(re_exp, str)
    if res:
        return True 
    else:
        return False
    
def set_all_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)