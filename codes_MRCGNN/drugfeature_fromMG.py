
import os
import torch
import random

import copy


import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=False)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for k in range(1): # modified, used to be 5
    os.chdir('/fs/student/jianyu_hou/cs292F/MRCGNN/codes_MRCGNN/trimnet/')
    cmd = (
        "python train.py "
        "--zhongzi " + str(k) + " "
        "--n_epochs 100 " 
        "--lr 0.001 " 
        "--batch_size 128 " 
    )
    os.system(cmd)