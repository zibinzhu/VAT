import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torchvision.utils as vutils
import torch, random
import torch.nn.functional as F

from tools import test_pifu
random.seed(1991)
np.random.seed(1991)
torch.manual_seed(1991)
torch.cuda.manual_seed(1991)
test_pifu.main()