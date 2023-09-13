#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 07:18:10 2022

@author: forootani
"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DataSetFod(Dataset):

    """
    Discription:
        a class that inherits from Datset torch module and prepare the
        data set for us
    attributes:
        time: [t_min:step size :t_max]
        output: dynamic system outputs, e.g. x_1, x_2, etc.
        initial_conds: initial conditions
        length: length of time span [t_min:step size :t_max]
        device: CPU/GPU

    only __getitem__() method is implmented which call by index
    the other can be written if it is required

    It is read-able only and gives: input and output of our DNN module

    """

    def __init__(self, time, initial_conds, output, device):
        self.time = time
        self.output = output
        self.initial_conds = initial_conds
        self.length = time.size()[0]
        self.device = device

    def __getitem__(self, index):
        current_time_init_conds = torch.cat((self.time, self.initial_conds), -1)
        current_input = current_time_init_conds[index]
        current_time = self.time[index]
        current_output = self.output[index]

        return current_input, current_output

    def __len__(self):
        return self.length

    def device_type(self):
        return self.device


def train_test_spliting_dataset(
    time, init_conds, output, device, batch_size, split_value, shuffle
):
    """
    Discription:
        a function that takes args and creat an instance of DataSetFod class
        to prepare the dataset

    args:
        time: [t_min:step size :t_max]
        init_conds: initial conditions
        device: CPU/GPU
        batch_size: int, e.g. 2000
        split_value: float, 0.9, splitting the data set for training and testing
        shuffle: bool, True/False, shuffling our dataset or not
    
    return:
        
        main_train_dataloader
        main_test_dataloader

    """

    length_data = len(time[0])

    main_train_dataloader = []
    main_test_dataloader = []

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

        train_dataset = DataSetFod(time_train, init_conds_train, output_train, device)

        test_dataset = DataSetFod(time_test, init_conds_test, output_test, device)

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

    return main_train_dataloader, main_test_dataloader
