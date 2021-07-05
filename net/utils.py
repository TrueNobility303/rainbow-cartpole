import torch 
import numpy as np 

def initial_seed(seed):
    np.random.seed(seed)
    #env.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True