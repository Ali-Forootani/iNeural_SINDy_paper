#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:33:57 2022

@author: forootani
"""
from itertools import chain,combinations, combinations_with_replacement as combinations_w_r
from functools import reduce
from typing import List, NewType
import numpy as np
import torch
from torch.autograd import grad
from typing import Tuple
from Functions.root_classes import Library

TensorList = NewType("TensorList", List[torch.Tensor])

# ==================== Library helper functions =======================

def library_poly(prediction: torch.Tensor, max_order: int) -> torch.Tensor:
    """
    A function to compute different combination of a given single column input data,
    x -> [x, x*x, x*x*x, ... ]
    We did not use this function here
    """
    u = prediction
    
    for order in np.arange(1, max_order):
        u = torch.cat((u, u[:, order - 1 : order] * prediction), dim=1)
    return u

################################
################################
def library_deriv(
    data: torch.Tensor, prediction: torch.Tensor, time_deriv_coef
) -> torch.Tensor:
    
    """
        Time derivative: Computing the derivative of the output with respect to the input
        time --- left handside of the dynamic system e.g. dx/dt = f(x)
    Args:
        data: which in our case is time, single column
        prediction: NN output, e.g. x, then we need to compute dx/dt
    """
    dy = grad(
        prediction, data, grad_outputs = torch.ones_like(prediction),
        create_graph=True)[0]
    time_deriv =  time_deriv_coef * dy[:, 0:1]    
    return time_deriv
#################
#################
#################
def _combinations(n_features, degree, include_interaction, include_bias,
                  interaction_only):  
    """ 
    Finding the number of permutaion to construc the library
    Args: n_features, degree, include_interaction, include_bias,
                  interaction_only 
        n_features: number of variables in the differential equation, e.g. x,y,z
        degree: polynomial degree
        include_interaction : boolian, True is we need interaction terms in the library terms, else False
        include_bias: boolian, False if we need interaction terms in the library terms, else False
        interaction_only: boolian, True if we need intraction terms in the library or not, else False 
    """
    comb = combinations if interaction_only else combinations_w_r
    start = int(not include_bias)
    if not include_interaction:
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
    """
    Making polynomial library
    Args:
        x: in our case the output of NN module which is prediction
        degree: degree polynomial
        include bias: boolian
        include interaction: boolian
        interaction_only: boolian
    """
    n_samples, n_features = x.shape
    bias = torch.reshape(torch.pow(x[:,0],0),(n_samples,1))
    to_stack = []
    if include_bias:
        to_stack.append(bias)
    combinations_main =_combinations(n_features, degree, include_interaction,
                                     include_bias, interaction_only)
    n_output_features_ = sum(1 for _ in combinations_main)
    combinations_main_2 =_combinations(n_features, degree, include_interaction,
                                       include_bias, interaction_only)
    columns=[]
    for i in combinations_main_2:
        if i:
            out_col = 1
            for col_idx in i:
                out_col = x[:, col_idx].multiply(out_col)

            out_col = torch.reshape(out_col,(n_samples,1))
            columns.append(out_col)
        else:
            bias = torch.reshape(torch.pow(x[:,0],0),(n_samples,1))
            columns.append(bias)
    thetas = torch.t(torch.stack(columns).squeeze())
    xp=torch.reshape(thetas,(n_samples,n_output_features_))
    return thetas
#################
#################
#################
# ========================= Actual library functions ========================

class LibraryObject(Library):
    """A class to make a library instance which inhereits from Library calss in root_class:
        Arges:
                ploy_order: maximum order of the terms, e.g. if poly_order=2, then [1, x, y, xx, xy, yy]
                include_interaction: boolian, to include interaction terms in the library or not, like 'xy'
                include_bias: boolian, to include bias term or not, like '1'
                interaction_only: boolian, to condier only intraction terms in the library, like 'xy'
                time_deriv_coef: this a coefficient which corrects the time derivative due to scaling the time in the interval
                                    [-1, 1], this coefficient cross out the effect of sacling so the training of the NN is smoother. 
     """
    def __init__(self, poly_order: int, include_interaction, include_bias,
                      interaction_only, time_deriv_coef) -> None:
        super().__init__()
        self.poly_order = poly_order
        self.include_interaction = include_interaction
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.time_deriv_coef = time_deriv_coef
    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ):
        """
        The main function to creat the polynomial library
            'prediction' and 'data' are inherited from parent class Libray in the root_class
        """
        
        prediction, data = input
        #poly_list = []
        #deriv_list = []
        time_deriv_list = []
        
        n_features = prediction.shape[1]
        
        combinations_main = _combinations(n_features, self.poly_order,
                                          self.include_interaction,
                                          self.include_bias,
                                          self.interaction_only)                                 
        # to see quickly the number of terms in the library we can write:
        #n_output_features_ = sum(1 for _ in combinations_main)
      # Creating lists for all outputs
        for output in np.arange(prediction.shape[1]):
            time_deriv = library_deriv(
                data, prediction[:, output : output + 1], self.time_deriv_coef )
            time_deriv_list.append(time_deriv)   
        time_deriv_list = torch.t(torch.squeeze(torch.stack(time_deriv_list)))
        theta = transform_torch(prediction, self.poly_order, self.include_interaction
                                , self.include_bias, self.interaction_only)
        return time_deriv_list, theta
