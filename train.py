import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from tools import train_pifu

train_pifu.main()
