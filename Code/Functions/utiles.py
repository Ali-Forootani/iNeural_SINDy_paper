#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:26:04 2022

@author: forootani
"""
from random import random, randrange
import numpy as np
import torch
import torch.nn as nn
import random as rnd
############################
############################

## Normalize data
class NormalizedData(nn.Module):
    """
    A class to normalize the input
    """
    def __init__(self, x: "numpy array"):
        super(NormalizedData, self).__init__()
        self.x = x
        self.mean = [np.mean(self.x[:, :, i]) for i in range(self.x.shape[2])]
        self.std = [np.std(self.x[:, :, i]) for i in range(self.x.shape[2])]
        # self.mean = [0.,0.,25.]
        # self.std = [8.,8.,8.]

    def normalize_meanstd(self):
        """
        A functon to normalize the input data
        """
        x_nor = np.zeros_like(self.x)
        for i in range(self.x.shape[2]):
            x_nor[:, :, i] = (self.x[:, :, i] - self.mean[i]) / self.std[i]
        return x_nor


############################
############################
def time_scaling_func(t: "torch Tensor"):
    """
    Discription: Sacling the time between the interval [-1,1]
    Args:
         t: torch tensor: time which is [t_min: time_step: t_max]
    """
    t_std = torch.div(t - torch.min(t), torch.max(t) - torch.min(t))
    t_scaled = torch.mul(t_std, 1 + 1) - 1
    coords = t_scaled.reshape(-1, 1)
    return coords


############################
############################
def initial_cond_generation(num_init_cond:int, num_indp_var:int, min_init_cond:float, max_init_cond:float):
    """
    A function to generate a list of random inital conditions
        Args:
            num_init_cond: number of initial condition, an integer
            num_indp_var: number of independent variable, an integer e.g. x,y
            min_init_cond: min value for initial condition, float
            max_init_cond: max value for initial condition, float
    """
    list_initial_conditions = []
    for i in range(int(num_init_cond)):
        element_init = []
        for j in range(int(num_indp_var)):
            element_init.append(
                random() * randrange(min_init_cond, max_init_cond) + min_init_cond
            )
        list_initial_conditions.append(element_init)

    return list_initial_conditions

#############################
#############################


def set_all_seeds(seed):
    """
    A function for ensuring reproducibility in machine learning experiments
    """
    rnd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
##############################
##############################    

class HeatmapSetting:
   
    """
    Discription:
    a class for heatmap setting
    
    return: a dictionary that we pass to sns.heatmap(**kwargs)
    
    This class provides a convenient way to set properties for Seaborn 
    heatmaps and convert them to a dictionary format that can be easily
    passed to the sns.heatmap function.
    
    The class has two attributes v_max and fontsize, which are used to set 
    the maximum value of the color scale and the font size of the 
    annotations on the heatmap, respectively.
    
    """
    
    
    def __init__(self, v_max, fontsize):
        self.linewidth = 0.5
        self.cmap = "crest"
        self.annot = True
        self.fmt = ".3f"
        self.square = True
        self.vmin = 0
        self.vmax = v_max
        self.annot_kws = {'fontsize': fontsize,'fontweight': "bold",}
        
    def __call__(self):
        output_dict = {}
        for k, v in self.__dict__.items():
            #print(k)
            if isinstance(v, dict):
                v = {k2: v2 for k2, v2 in v.items()}
            output_dict[k] = v
        return output_dict


