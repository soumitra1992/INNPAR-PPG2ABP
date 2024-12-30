import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


def Grad(x):
    kernelx = torch.FloatTensor([[-1,0,1]]).unsqueeze(0)
    gradx = F.conv1d(x,kernelx,padding = 1)
    return gradx
    
class ReconstructData(data.Dataset):
    def __init__(self, data_dir):
        self.dir_ppg = os.path.join(data_dir, 'ppg')
        self.dir_abp = os.path.join(data_dir, 'abp')
        self.name = os.listdir(self.dir_ppg)
        # print(self.name)
        
    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        ppg = np.load(os.path.join(self.dir_ppg,self.name[idx]),allow_pickle=True).astype(np.float32)
        abp = np.load(os.path.join(self.dir_abp,self.name[idx]),allow_pickle=True).astype(np.float32)
        ppg = torch.from_numpy(ppg).unsqueeze(0)
        abp = torch.from_numpy(abp).unsqueeze(0)
        ppg_grad = Grad(ppg)
        ppg = torch.cat((ppg,ppg_grad),dim=1).squeeze(0)
        abp_grad = Grad(abp)
        abp = torch.cat((abp,abp_grad),dim=1).squeeze(0)
        C,L = ppg.shape
        if L%4!=0:
            num=4-L%4
            for i in range(num):
                ppg = F.pad(ppg,pad=(0,1),mode='constant',value=0)
                abp = F.pad(abp,pad=(0,1),mode='constant',value=0)
        return ppg,abp