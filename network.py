import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import os
import time
import numpy as np



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  
                if m.bias is not None:
                    m.bias.data.zero_()        
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  
                if m.bias is not None:
                    m.bias.data.zero_()        
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape))[0]
        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.linalg.lu_factor(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)
            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye
        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed
    def get_weight(self, input, reverse):
        b, c, L = input.shape
        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * L
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            lower = self.lower * self.l_mask + self.eye
            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = torch.sum(self.log_s) * L
            if reverse:
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)
                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))
        return weight.view(self.w_shape[0], self.w_shape[1], 1), dlogdet
    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv1d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv1d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet
def squeeze1d(input, factor,length='odd'):
    if factor == 1:
        return input
    B, C, L = input.size()
    assert L % factor == 0 , "L modulo factor is not 0"
    x = input.view(B, C, L // factor, factor)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B, C * factor, L // factor)
    return x
def unsqueeze1d(input, factor):
    if factor == 1:
        return input
    B, C, L = input.size()
    assert C % (factor) == 0, "C module factor is not 0"
    x = input.view(B, C // factor, factor, L)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(B, C // (factor), L * factor)
    return x
class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze1d(input, self.factor)
        else:
            output = squeeze1d(input, self.factor)
        return output #, logdet
    
class UnSqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = squeeze1d(input, self.factor)
        else:
            output = unsqueeze1d(input, self.factor)
        return output #, logdet    
        
        
class MSCM(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,feats=16, kernel_size=5,init = 'xavier'):
        super(MSCM, self).__init__()
        
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, feats, kernel_size=kernel_size, stride=1, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.main = nn.Sequential(
                # encoder
                SqueezeLayer(2),
                nn.Conv1d(feats*2, feats*2, kernel_size=kernel_size, stride=1, padding='same'),
                SqueezeLayer(2),
                # bottole-neck
                nn.Conv1d(feats*4, feats*4, kernel_size=kernel_size, stride=1, padding='same'),
                # decoder
                UnSqueezeLayer(2),
                nn.Conv1d(feats*2, feats*2, kernel_size=kernel_size, stride=1, padding='same'),
                UnSqueezeLayer(2)
        )
        self.output_proj = nn.Sequential(
            nn.Conv1d(feats, out_ch, kernel_size=kernel_size, stride=1, padding='same')
        )
        
        if init == 'xavier':
            initialize_weights_xavier([self.input_proj, self.main, self.output_proj], 0.1)
        else:
            initialize_weights([self.input_proj, self.main, self.output_proj], 0.1)
            
    def forward(self, x):
        x = self.input_proj(x)
        x = self.main(x)
        x = self.output_proj(x)
        return x    
def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out,feats=16, kernel_size=5):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return MSCM(in_ch=channel_in,out_ch=channel_out,feats=feats, kernel_size=kernel_size, init=init)
            else:
                return MSCM(in_ch=channel_in,out_ch=channel_out,feats=feats, kernel_size=kernel_size)
        else:
            return None
    return constructor
class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num=2, channel_split_num=1, clamp=0.8,feats=16, kernel_size=5):
        super(InvBlock, self).__init__()
        # channel_num: 2
        # channel_split_num: 1
        self.split_len1 = channel_split_num # 1
        self.split_len2 = channel_num - channel_split_num # 1 
        self.clamp = clamp
        self.F = subnet_constructor(self.split_len2, self.split_len1,feats=feats, kernel_size=kernel_size)
        self.G = subnet_constructor(self.split_len1, self.split_len2,feats=feats, kernel_size=kernel_size)
        self.H = subnet_constructor(self.split_len1, self.split_len2,feats=feats, kernel_size=kernel_size)
        in_channels = channel_num        
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        
    def forward(self, x, rev=False):
        if not rev:            
            # invert1x1conv 
            x, logdet = self.flow_permutation(x, logdet=0, rev=False) 
            
            # split to 1 channel and 2 channel. 
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 
            y1 = x1 + self.F(x2) # 1 channel 
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1) # 2 channel 
            out = torch.cat((y1, y2), 1)
        else:
            # split. 
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)) 
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s)) 
            y1 = x1 - self.F(y2) 
            x = torch.cat((y1, y2), 1)            
            # inv permutation 
            out, logdet = self.flow_permutation(x, logdet=0, rev=True)
        return out
class INNPAR(nn.Module):
    def __init__(self, channel_in=2, channel_out=2, subnet_constructor=subnet('DBNet'), block_num=4,feats=16, kernel_size=5):
        super(INNPAR, self).__init__()
        operations = []
        current_channel = channel_in
        channel_num = channel_in
        channel_split_num = 1
        for j in range(block_num): 
            b = InvBlock(subnet_constructor, channel_num, channel_split_num,feats=feats, kernel_size=kernel_size) # one block is one flow step. 
            operations.append(b)
        
        self.operations = nn.ModuleList(operations)
        self.initialize()
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.  
                if m.bias is not None:
                    m.bias.data.zero_()         
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= 1.
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self, x, rev=False):
        out = x # x: [N,3,H,W] 
        
        if not rev: 
            for op in self.operations:
                out = op.forward(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
        
        return out
        
