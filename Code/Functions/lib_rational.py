#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:33:57 2022

@author: forootani
"""

import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations
from functools import reduce
from Functions.root_classes import Library
from typing import Tuple
from typing import List, NewType
from sklearn.preprocessing import PolynomialFeatures


from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

#import polynomial_library_torch as pl_torch

#from polynomial_library_torch import PolynomialLibrary


TensorList = NewType("TensorList", List[torch.Tensor])





# ==================== Library helper functions =======================


"""
polynomial library 
"""


def library_poly(prediction: torch.Tensor, max_order: int) -> torch.Tensor:
    
    u = prediction
    
    for order in np.arange(1, max_order):
        u = torch.cat((u, u[:, order - 1 : order] * prediction), dim=1)
    
    
    return u


"""
Time derivative: Computing the derivative of the output with respect to the
time --- left handside of the dynamic system e.g. dx/dt = f(x)

Args:
    data: which in our case is time, single column
    prediction: NN output 

"""

    
def library_deriv(
    data: torch.Tensor, prediction: torch.Tensor, time_deriv_coef
) -> torch.Tensor:
    
    
    dy = grad(
        prediction, data, grad_outputs = torch.ones_like(prediction),
        create_graph=True)[0]
    
    
    time_deriv =  time_deriv_coef * dy[:, 0:1]
    
    #time_deriv =  1 * dy[:, 0:1]

    
    return time_deriv


#################
#################
#################

""" 
Finding the number of permutaion to construc the library --- 
Args: n_features, degree, include_interaction, include_bias,
                  interaction_only 
    n_features: number of variables in the differential equation, e.g. x,y,z
    degree: polynomial degree
    include_interaction : boolian, True is we need interaction terms in the library terms, else False
    include_bias: boolian, False if we need interaction terms in the library terms, else False
    interaction_only: boolian, True if we need intraction terms in the library or not, else False 
"""



def _combinations(n_features, degree, include_interaction, include_bias,
                  interaction_only):  
    
    ""
    
    comb = combinations if interaction_only else combinations_w_r
    
    start = int(not include_bias)    
    
    
    if not include_interaction:
        
        #print("I am here my friend")
        
        if include_bias:
            return chain(
                [()],
                chain.from_iterable(
                    combinations_w_r([j], i)
                    for i in range(1, degree + 1)
                    for j in range(n_features)
                ),
            )
        else:
            return chain.from_iterable(
                combinations_w_r([j], i)
                for i in range(1, degree + 1)
                for j in range(n_features)
            )
        
    return chain.from_iterable(
        comb(range(n_features), i) for i in range(start, degree + 1)
    )

#################
#################
#################


def transform_torch(x,degree,include_interaction, include_bias,
                  interaction_only):
    
    n_samples, n_features = x.shape
    
    
    bias = torch.reshape(torch.pow(x[:,0],0),(n_samples,1))
    
    
    to_stack = []
    
    
    if include_bias:
        to_stack.append(bias)
        #to_stack = torch.cat((bias, x), dim= -1)
        
        #torch.cat((torch.ones_like(samples), theta_uv),dim = 1)
    
    
    
    combinations_main =_combinations(n_features, degree, include_interaction,
                                     include_bias, interaction_only)
    
    
    n_output_features_ = sum(1 for _ in combinations_main)
    
    
    combinations_main_2 =_combinations(n_features, degree, include_interaction,
                                       include_bias, interaction_only)
    
    columns=[]
    
    for i in combinations_main_2:
        
        if i:
            
            #print(i)
            
            out_col = 1
            for col_idx in i:
                out_col = x[:, col_idx].multiply(out_col)
                
            out_col = torch.reshape(out_col,(n_samples,1))
            columns.append(out_col)
        else:
            #bias = sparse.csc_matrix(torch.ones((x.shape[0], 1)))
            bias = torch.reshape(torch.pow(x[:,0],0),(n_samples,1))
            columns.append(bias)
     
    thetas = torch.t(torch.stack(columns).squeeze())
    
    
    
    
    xp=torch.reshape(thetas,(n_samples,n_output_features_))
    
   
    return thetas


#################
#################
#################







# ========================= Actual library functions ========================

class library_first_ord_eq(Library):
    def __init__(self, poly_order: int, include_interaction, include_bias,
                      interaction_only, time_deriv_coef) -> None:
        

        super().__init__()
        self.poly_order = poly_order
        self.include_interaction = include_interaction
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.time_deriv_coef = time_deriv_coef
        
        #self.diff_order = diff_order

    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ):
        
        prediction, data = input
        poly_list = []
        #deriv_list = []
        time_deriv_list = []
        
        n_features = prediction.shape[1]
        
        combinations_main = _combinations(n_features, self.poly_order,
                                          self.include_interaction,
                                          self.include_bias,
                                          self.interaction_only)
        n_output_features_ = sum(1 for _ in combinations_main)
        
        
        
        
        
      # Creating lists for all outputs
        for output in np.arange(prediction.shape[1]):
            
           
            
            time_deriv = library_deriv(
                data, prediction[:, output : output + 1], self.time_deriv_coef )
            
            
            time_deriv_list.append(time_deriv)
            
            
            
           
        time_deriv_list = torch.t(torch.squeeze(torch.stack(time_deriv_list)))
        
        
        
        
        theta = transform_torch(prediction, self.poly_order, self.include_interaction
                                , self.include_bias, self.interaction_only)
        
        
        
        return time_deriv_list, theta
