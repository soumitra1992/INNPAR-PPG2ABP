import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import *
from network import *
from data import *
from permetrics.regression import RegressionMetric
import warnings
 
warnings.filterwarnings('ignore')

manual_seed=1
num_workers=12
lr = 1e-4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")          

def run():
    checkpoint_dir='./checkpoints'
    seed_everything(manual_seed)
    model=INNPAR().to(device)
    PATH=os.path.join(checkpoint_dir, 'INNPAR.pth')
    print(PATH)
    
    model.load_state_dict(torch.load(PATH))
    
    
    data_dir='./datasets/test/sensors'
    testloader = DataLoader(ReconstructData(data_dir),batch_size=1,shuffle=False,num_workers =num_workers,pin_memory=True)
    nrmse = 0.0
    mae = 0.0
    for ind,(ppg,abp) in enumerate(testloader):
        ppg,abp = ppg.to(device), abp.to(device)
        model.eval()
        with torch.no_grad():
            x = model(ppg)
            xs = x[:,0,0:625]
            abps = abp[:,0,0:625]
            abp_np =np.squeeze(np.array(abps.cpu()))
            x_np =np.squeeze(np.array(xs.cpu()))
            evaluator = RegressionMetric(abp_np, x_np)
            nrmse += evaluator.normalized_root_mean_square_error()
            mae += evaluator.mean_absolute_error()
    test_loss1 = nrmse/len(testloader.sampler)
    test_loss2 = mae/len(testloader.sampler)
    print("Sensors dataset:")
    print("NRMSE loss: ",test_loss1)
    print("MAE loss: ",test_loss2)
    
        
    data_dir='./datasets/test/bcg'
    testloader = DataLoader(ReconstructData(data_dir),batch_size=1,shuffle=False,num_workers =num_workers,pin_memory=True)
    nrmse = 0.0
    mae = 0.0
    for ind,(ppg,abp) in enumerate(testloader):
        ppg,abp = ppg.to(device), abp.to(device)
        model.eval()
        with torch.no_grad():
            x = model(ppg)
            xs = x[:,0,0:625]
            abps = abp[:,0,0:625]
            abp_np =np.squeeze(np.array(abps.cpu()))
            x_np =np.squeeze(np.array(xs.cpu()))
            evaluator = RegressionMetric(abp_np, x_np)
            nrmse += evaluator.normalized_root_mean_square_error()
            mae += evaluator.mean_absolute_error()
    test_loss3 = nrmse/len(testloader.sampler)
    test_loss4 = mae/len(testloader.sampler)
    print("BCG dataset:")
    print("NRMSE loss: ",test_loss3)
    print("MAE loss: ",test_loss4)
        
if __name__ == '__main__':
    run()
