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
from root_classes import Library
from typing import Tuple
from typing import List, NewType

from sklearn.preprocessing import PolynomialFeatures


from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r





TensorList = NewType("TensorList", List[torch.Tensor])





# ==================== Library helper functions =======================
def library_poly(prediction: torch.Tensor, max_order: int) -> torch.Tensor:
    
    
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order + 1):
        u = torch.cat((u, u[:, order - 1 : order] * prediction), dim=1)
    
    
    return u


    
def library_deriv(
    data: torch.Tensor, prediction: torch.Tensor
) -> torch.Tensor:
        
    dy = grad(
        prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True
    )[0]
    
    
    time_deriv = dy[:, 0:1]
    
    
    return time_deriv


#################
#################
#################



def _combinations(n_features, degree, include_interaction=False, include_bias=True,
                  interaction_only=False):  
    
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
    
    #print("number of features*************")
    #print(n_output_features_)
    
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








#################
#################
#################


# ========================= Actual library functions ========================
class library_first_ord_eq(Library):
    def __init__(self, poly_order: int) -> None:
        

        super().__init__()
        self.poly_order = poly_order
        #self.diff_order = diff_order

    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ):
        
        prediction, data = input
        poly_list = []
        #deriv_list = []
        time_deriv_list = []
        
	
        # Creating lists for all outputs
        for output in np.arange(prediction.shape[1]):
            time_deriv = library_deriv(
                data, prediction[:, output : output + 1] )
            
            
            u = library_poly(prediction[:, output : output + 1], self.poly_order)

            poly_list.append(u)
            
            time_deriv_list.append(time_deriv)
            
            #deriv_list.append(du)
            

        #samples = time_deriv_list[0].shape[0]
        
        samples = time_deriv_list[0].shape[0]
        total_terms = poly_list[0].shape[1] 
        
        
        # Calculating theta
        if len(poly_list) == 1:
            
            theta = torch.tensor(poly_list[0][:,:]).view(poly_list[0].shape[0]
                                                         ,poly_list[0].shape[1])
        
            
        theta = transform_torch(prediction, self.poly_order, True
                               , True, False)     
            
        
        #include_interaction=True, include_bias=True,
        #                  interaction_only=False
        
        #print("================8888888888888888")
        #print("================8888888888888888")
        #print("================8888888888888888")

        #print(theta_num)
        
        #print(theta)
        
        #print("================8888888888888888")
        #print("================8888888888888888")
        #print("================8888888888888888")

        
        
        return time_deriv, theta
