#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:49:33 2022

@author: forootani
"""

import torch
import torch.nn as nn
from Functions.lib_rational import transform_torch


## Define coeffis for a dictionary
class CoeffsNetwork(nn.Module):
    """
    Defining the sparse coefficiets and in the forward pass,
    we obtain multiplication of features and sparse coefficients.
    ----------
    n_combinations : int: the number of features in dictionary
        DESCRIPTION.
    n_features : int : the number of variables
        DESCRIPTION.
    
    torch.nn.init.xavier_uniform(conv1.weight)
    
    Returns
    -------
    Product of features multiplied by sparse coefficients.
    """

    def __init__(self, n_combinations, n_features, zero_inits=True):
        super().__init__()
        self.linear = nn.Linear(n_combinations, n_features, bias=False)
        self.zero_inits = zero_inits
        # Setting the weights to zeros
        if self.zero_inits is True:
            self.linear.weight = torch.nn.Parameter(0 * self.linear.weight.clone().detach())
        else:
            self.linear.weight = torch.nn.Parameter(1 * self.linear.weight.clone().detach())
            #self.linear.weight = torch.nn.init.xavier_uniform(self.linear.weight.clone().detach())
            #torch.nn.Parameter(0 * self.linear.weight.clone().detach())
            #torch.nn.init.xavier_uniform(conv1.weight)
        

    def forward(self, x):
        
        return self.linear(x.float())


class CoeffsDictionaryRational(nn.Module):
    """
    Defining the sparse coefficiets and in the forward pass,
        we obtain a ratio of multiplications of features and sparse coefficients.
        ----------
        n_combinations : int: the number of features in dictionary
            DESCRIPTION.
        n_features : int : the number of variables
            DESCRIPTION.
        Returns
        -------
        Product of features multiplied by sparse coefficients.
    """

    def __init__(self, n_combinations, n_features):
        super(CoeffsDictionaryRational, self).__init__()
        self.numerator = nn.Linear(n_combinations, n_features, bias=False)
        self.denominator = nn.Linear(n_combinations - 1, n_features, bias=False)

        # Setting weights to zero
        self.numerator.weight = torch.nn.Parameter(
            0 * self.numerator.weight.clone().detach()
        )
        self.denominator.weight = torch.nn.Parameter(
            (0 * self.denominator.weight.clone().detach())
        )

    def forward(self, x):
        N1 = self.numerator(x.float())
        D1 = self.denominator(x[:, 1:].float())
        return N1 / (D1 + 1)


## Simple RK model
def rk4th_onestep(model, x, t=0, timestep=1e-2):
    """
    This is a simple Runge-Kutta scheme which is applied on NN model

    Borrowed from the paper: 'Discovery of Nonlinear Dynamical Systems using
    a Runge-Kutta Inspired Dictionary-based Sparse Regression Approach'
    Args:
         model: NN model defined earlier
         x: torch tensor: in our case is the data
         t: torch tensor: time
         timestep: float value
    """
    k_1 = model(x, t)
    k_2 = model(x + 0.5 * timestep * k_1, t + 0.5 * timestep)
    k_3 = model(x + 0.5 * timestep * k_2, t + 0.5 * timestep)
    k_4 = model(x + 1.0 * timestep * k_3, t + 1.0 * timestep)
    return x + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * timestep


## Simple RK-SINDy model    ####### timestep is based on the length of time horizon
## timestep = (t_f - t_s)/ num_samples
def rk4th_onestep_SparseId(x, library, library_k, model, t=0, timestep=0.025):
    """
    Description:
    Applying RK4 on rational dynamic systems  k(x)+ N(x)/(D(x)+1)
    for the case of non-rational dynamic systems we can use k(x)
    Args:
        x: torch tensor: data
        library: torch tensor: library for the term N(x)/(D(x)+1)
        library_k: torch tensor: library for the term k(x)
        model: nn.module: NN network that we defined earlier
        t: torch tensor: time in our case
        timestep: float: t(k)-t(k-1)
    """

    theta_k = transform_torch(
        x,
        library_k.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=False,
    )
    theta_N_D = transform_torch(
        x,
        library.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=True,
    )

    Num = theta_N_D.float() @ torch.t(model.estimated_coeffs.numerator.weight)
    Den = theta_N_D[:, 1:].float() @ torch.t(model.estimated_coeffs.denominator.weight)
    K_X = theta_k.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k_1 = Num / (1 + Den) + K_X
    #############################################
    theta_k_2 = transform_torch(
        x + 0.5 * timestep * k_1,
        library_k.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=False,
    )
    theta_N_D_2 = transform_torch(
        x + 0.5 * timestep * k_1,
        library.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=True,
    )
    Num_2 = theta_N_D_2.float() @ torch.t(model.estimated_coeffs.numerator.weight)
    Den_2 = theta_N_D_2[:, 1:].float() @ torch.t(
        model.estimated_coeffs.denominator.weight
    )
    K_X_2 = theta_k_2.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k_2 = Num_2 / (1 + Den_2) + K_X_2
    # k1 = LibsCoeffs(d1)
    # d2 = library.transform_torch(x + 0.5* timestep* k1, )
    theta_k_3 = transform_torch(
        x + 0.5 * timestep * k_2,
        library_k.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=False,
    )
    theta_N_D_3 = transform_torch(
        x + 0.5 * timestep * k_2,
        library.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=True,
    )
    Num_3 = theta_N_D_3.float() @ torch.t(model.estimated_coeffs.numerator.weight)
    Den_3 = theta_N_D_3[:, 1:].float() @ torch.t(
        model.estimated_coeffs.denominator.weight
    )
    K_X_3 = theta_k_3.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k_3 = Num_3 / (1 + Den_3) + K_X_3
    # k2 = LibsCoeffs(d2)
    # d3 = library.transform_torch(x + 0.5* timestep* k2)
    # k3 = LibsCoeffs(d3)
    theta_k_4 = transform_torch(
        x + 1 * timestep * k_3,
        library_k.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=False,
    )
    theta_N_D_4 = transform_torch(
        x + 1 * timestep * k_3,
        library.poly_order,
        include_interaction=False,
        include_bias=True,
        interaction_only=True,
    )
    Num_4 = theta_N_D_4.float() @ torch.t(model.estimated_coeffs.numerator.weight)
    Den_4 = theta_N_D_4[:, 1:].float() @ torch.t(
        model.estimated_coeffs.denominator.weight
    )
    K_X_4 = theta_k_4.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k_4 = Num_4 / (1 + Den_4) + K_X_4
    # d4 = library.transform_torch(x + 1.0* timestep* k3)
    # k4 = LibsCoeffs(d4)
    return x + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * timestep


def rk4th_onestep_SparseId_non_rational(x, library, model, timestep, t=0):
    """
    Description:
    Applying RK4 on non-rational dynamic systems  k(x)+
    Args:
        x: torch tensor: data
        library: torch tensor: library for the term k(x)
        model: nn.module: NN network that we defined earlier
        t: torch tensor: time in our case
        timestep: float: t(k)-t(k-1)
    """

    poly_dic = transform_torch(
        x,
        library.poly_order,
        include_interaction=True,
        include_bias=True,
        interaction_only=False,
    )
    # k1= theta.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k1 = model.estimated_coeffs_k(poly_dic)
    # k1 = model.estimated_coeffs_k(theta)
    poly_dic2 = transform_torch(
        x + 0.5 * timestep * k1,
        library.poly_order,
        include_interaction=True,
        include_bias=True,
        interaction_only=False,
    )
    # k2 = theta2.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k2 = model.estimated_coeffs_k(poly_dic2)
    # k2 = model.estimated_coeffs_k(theta2)
    poly_dic3 = transform_torch(
        x + 0.5 * timestep * k2,
        library.poly_order,
        include_interaction=True,
        include_bias=True,
        interaction_only=False,
    )
    # k3 = theta3.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k3 = model.estimated_coeffs_k(poly_dic3)
    poly_dic4 = transform_torch(
        x + 1.0 * timestep * k3,
        library.poly_order,
        include_interaction=True,
        include_bias=True,
        interaction_only=False,
    )
    # k4 = theta4.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    k4 = model.estimated_coeffs_k(poly_dic4)
    # x + (1/6)*(k1+2*k2+2*k3+k4)*timestep
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep
