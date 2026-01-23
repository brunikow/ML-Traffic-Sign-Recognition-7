import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """
    Initiates a global seed for reproducibility. 
    
    @param seed: the seed to be used (default: 42).
    """   
    random.seed(seed)    
    np.random.seed(seed)    
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True 

    print(f"global seed set: {seed}")

def seed_worker(worker_id: int) -> None:
    """
    Initiates seed for workers in DataLoader.  
    
    @param worker_id: ID num_workers
    """
    worker_seed = torch.initial_seed() % 2**32 # to fit pytorch into numpy use modulo
    np.random.seed(worker_seed)
    random.seed(worker_seed)