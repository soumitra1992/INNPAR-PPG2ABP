import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import os
import sys
import time
import datetime
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import *
from network import *
from data import *
from losses import *
import warnings
warnings.filterwarnings('ignore')
manual_seed=1
num_workers=12
lr = 1e-4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
@click.command()
@click.option('--max_step',default=500, help='No. of epochs')           

def run(max_step):
    data_dir='./datasets/train'
    loss_fn = model_loss() 
    batch_size = 128
    latest_subdir='./exp'
    log_dir = os.path.join(latest_subdir, 'tensorboard')
    checkpoint_dir=os.path.join(latest_subdir, 'checkpoint')
    results_dir=os.path.join(latest_subdir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)        
    file_writer = SummaryWriter(log_dir)
    seed_everything(manual_seed)
    model=INNPAR().to(device)
    trainloader = DataLoader(ReconstructData(data_dir),batch_size=batch_size,shuffle=True,num_workers =num_workers,pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(max_step+1):
        train_loss = 0.0
        for _,(ppg,abp) in enumerate(trainloader):
            ppg,abp = ppg.to(device), abp.to(device)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            abp_re = model(ppg)
            loss = loss_fn(abp_re,abp)
            loss.backward()
            optimizer.step() 
            train_loss += loss.item() * ppg.size(0)       
        
        train_loss = train_loss/len(trainloader.sampler)
        print('[Epoch:%d\tTraining Loss:%.3f]'%(epoch,train_loss))
        file_writer.add_scalar('Loss/train', train_loss, epoch)
        # Save the weight per 10 epoch
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_%s.pth' %(str(epoch))))
    
    
    with open(os.path.join(results_dir, 'training_result.txt'), 'w') as txt_file:
        txt_file.write("Model no. of parameters: " + str(no_parameters)+"\n")
        txt_file.write("No. of epochs: " + str(max_step)+"\n")
        txt_file.write("Training loss: " + str(train_loss)+"\n")
        
if __name__ == '__main__':
    run()
