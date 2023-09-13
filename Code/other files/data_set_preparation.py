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
    
    def __init__(self,time,initial_conds,output,device):
        self.time = time
        self.output = output
        self.initial_conds = initial_conds
        self.length = time.size()[0]
        self.device = device
        
    def __getitem__(self, index):
        current_time_init_conds = torch.cat((self.time,self.initial_conds),-1)
        current_input = current_time_init_conds[index]
        current_time = self.time[index]
        current_output = self.output[index]
        
        return current_input, current_output
    
    def __len__(self):
        
        return self.length
    
    def device_type(self):
        
        return self.device 




def train_test_spliting_dataset(time, init_conds, output,device,batch_size,split_value,shuffle):
    
    length_data = len(time[0])
    
    main_train_dataloader=[]
    main_test_dataloader=[]
    
    
    for i in range(len(time)):
        indices = np.arange(0, length_data, dtype=int)
        
        split = int(split_value * length_data)
        
        train_indices = indices[:split]
        
        test_indices = indices[split:]
        
        time_train = time[i][:split]
        time_test = time[i][split:]
        
        output_train = output[i][:split]
        output_test = output[i][split:]
        
        init_conds_train = init_conds[i][:split]
        init_conds_test = init_conds[i][split:]
        
        train_dataset = data_set_fod(time_train,init_conds_train,output_train,device)
        
        test_dataset = data_set_fod(time_test,init_conds_test,output_test,device)
        
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle)
        
        main_train_dataloader.append(train_dataloader)
        main_test_dataloader.append(test_dataloader)
        
        
    
        
    """ old code that uses the whole data as a pack
    
        
    length = len(time)
    
    
    #split_value=0.9
    
    indices = np.arange(0, length, dtype=int)
    
    split = int(split_value * length)
    
    train_indices = indices[:split]
    
    test_indices = indices[split:]
    
    time_train = time[:split]
    time_test = time[split:]
    
    
    
    output_train = output[:split]
    output_test = output[split:]
    
    init_conds_train = init_conds[:split]
    init_conds_test = init_conds[split:]
    
    
    
    train_dataset = data_set_fod(time_train,init_conds_train,output_train,device)
    
    test_dataset = data_set_fod(time_test,init_conds_test,output_test,device)
    
    
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle)
    
    #print(train_dataloader)
    #print(test_dataloader)
    
    """
    #print("8888888888888888888")
    #print("8888888888888888888")
    #print(main_train_dataloader)
    #print("8888888888888888888")
    #print("8888888888888888888")

    
    return main_train_dataloader, main_test_dataloader
