#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:26:04 2022

@author: forootani
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


from random import random,randrange


## Normalize data
class normalized_data(nn.Module):
    def __init__(self, x):
        super(normalized_data, self).__init__()
        self.x = x
        self.mean = [np.mean(self.x[:,:,i]) for i in range(self.x.shape[2])]
        self.std = [np.std(self.x[:,:,i]) for i in range(self.x.shape[2])]
        #self.mean = [0.,0.,25.]
        #self.std = [8.,8.,8.]
        
        
    def normalize_meanstd(self):
        x_nor = np.zeros_like(self.x)
        for i in range(self.x.shape[2]):
            x_nor[:,:,i] = (self.x[:,:,i] - self.mean[i])/self.std[i]
        return x_nor
    
    
def time_scaling_func(t):
    
    
    t_std = torch.div(t - torch.min(t),torch.max(t) - torch.min(t) )
    
    t_scaled = torch.mul(t_std, 1 + 1) - 1
    coords = t_scaled.reshape(-1,1)
    
    return coords



def initial_cond_generation(num_init_cond, num_indp_var, min_init_cond, max_init_cond):
    list_initial_conditions=[]
    for i in range(int(num_init_cond)):
        element_init=[]
        for j in range(int(num_indp_var)):
            element_init.append(random()*randrange(min_init_cond, max_init_cond))
            list_initial_conditions.append(element_init)
            
    return list_initial_conditions