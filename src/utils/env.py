import os

import torch
from pytorch_lightning import seed_everything


def setup_env(env):
     
    setup_torch()
    
    if "seed" in env.keys():
        print(f"Seed everything with <{env.seed}>")
        seed_everything(env.seed, workers=True)
    else:
        print(f"No seed is given.")

    if "max_threads" in env.keys():
        print(f"Set max threads to <{env.max_threads}>")
        set_max_threads(env.max_threads)
    else:
        print(f"Set max threads to autmatic")

def set_max_threads(max_threads):
    """Manually set max threads.

    Threads set up for:
    - OMP_NUM_THREADS
    - OPENBLAS_NUM_THREADS
    - MKL_NUM_THREADS
    - VECLIB_MAXIMUM_THREADS
    - NUMEXPR_NUM_THREADS
    """
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)
    torch.set_num_threads(max_threads)

def setup_torch():

    torch.set_float32_matmul_precision("high")
