#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 07:18:10 2022

@author: forootani
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class data_set_fod(Dataset):
    
    def __init__(self,time,output,device):
        self.time = time
        self.output = output
        self.length = time.size()[0]
        self.device = device
        
    def __getitem__(self, index):
        
        current_time = self.time[index]
        current_output = self.output[index]
        
        return current_time,current_output
    
    def __len__(self):
        
        return self.length
    
    def device_type(self):
        
        return self.device 


def train_test_spliting_dataset(time, output, device, batch_size):
    
    length = len(time)
    
    
    split_value=0.8
    
    indices = np.arange(0, length, dtype=int)
    
    split = int(split_value * length)
    
    train_indices = indices[:split]
    
    test_indices = indices[split:]
    
    time_train = time[:split]
    time_test = time[split:]
    output_train = output[:split]
    output_test = output[split:]
    
    train_dataset = data_set_fod(time_train,output_train,device)
    
    test_dataset = data_set_fod(time_test,output_test,device)
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader
