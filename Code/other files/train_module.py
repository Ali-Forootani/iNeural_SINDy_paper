#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 08:21:34 2022

@author: forootani
"""

import torch

from root_classes import complete_network



def train_func(train_dataloader, model, loss_fn, optimizer, epochs,device):

    
    for epoch in range (epochs):
        size = len(train_dataloader.dataset)
        
        print("====")
        print(size)
        print("====")
    
        model.train()
        for batch, (x,y) in enumerate(train_dataloader):
        
            x, y = x.to(device), y.to(device)
        
            #print(x)
            #print("***")
            #print(y)
        
            # Compute prediction error
            pred = model(x)
            
            
            #print("====")
            #print(len(pred[0]))
            #print(y)
            #print("====")
        
            loss = loss_fn(pred[0], y)
        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
            if batch % 20 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    

