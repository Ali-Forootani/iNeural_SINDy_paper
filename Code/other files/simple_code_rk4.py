#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:34:39 2022

@author: forootani
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

from lib_rational import transform_torch


 
def quad_2D(t, x):
 return [
 -0.1 * x[0] + 2 * x[1] - 0.5* x[0]**2,
 -2 * x[0] - 0.1 * x[1] + 0.25 * x[0]*x[1]
 ]

def linear_2D(t, x):
 return [
 -0.1 * x[0] + 2 * x[1],
 -2 * x[0] - 0.1 * x[1]
 ]



dynModel = linear_2D

dynModel = quad_2D

timestep = 1e-2
time_final = 10 
ts = np.arange(0,time_final,timestep)

# Initial condition and simulation time
x0 = [2,0]

# x = sol
sol = solve_ivp(dynModel, [ts[0], ts[-1]], x0, t_eval=ts)
# x = sol
x = np.transpose(sol.y)

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)

ax.plot(ts,x, linewidth = 2.5,label="Clean data")
ax.set(xlabel = 'time', ylabel = '$\{x_1, x_2\}$')

######################


def rk4th_onestep_SparseId_non_rational(x,model,timestep,t=0):
    
    
    
    
    poly_dic = transform_torch(x,2, include_interaction=True,
                                   include_bias=True,
                                   interaction_only=False)
    
    # k1= theta.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    
    k1 = model(poly_dic)
    
    
    
    #k1 = model.estimated_coeffs_k(theta)
    
    
    poly_dic2 = transform_torch(x + 0.5* timestep* k1,2, include_interaction=True,
                                   include_bias=True,
                                   interaction_only=False)
    # k2 = theta2.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    
    k2 = model(poly_dic2)
    
   
    
    
    #k2 = model.estimated_coeffs_k(theta2)
    
    poly_dic3 = transform_torch(x + 0.5* timestep* k2,2,include_interaction=True,
                                   include_bias=True,
                                   interaction_only=False)
    
    # k3 = theta3.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    
    k3 = model(poly_dic3)
    
    poly_dic4 = transform_torch(x + 1.0* timestep* k3,2,include_interaction=True,
                                   include_bias=True,
                                   interaction_only=False)
    
    #k4 = theta4.float() @ torch.t(model.estimated_coeffs_k.linear.weight)
    
    k4 = model(poly_dic4)
    
    
    #x + (1/6)*(k1+2*k2+2*k3+k4)*timestep
        
    return x + (1/6)*(k1+2*k2+2*k3+k4)*timestep




######################
class coeffs_network(nn.Module):
    def __init__(self, n_combinations=6, n_features=2):
        
        
        super().__init__()
        self.linear = nn.Linear(n_combinations,n_features,bias=False)
        
        # Setting the weights to zeros
        self.linear.weight = torch.nn.Parameter(0 * self.linear.weight.clone().detach())
        
    def forward(self,x):
        return self.linear(x)



########################
########################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

dynmodel = coeffs_network().to(device).double()

u = torch.tensor(x, dtype = torch.double).to(device)

optim = torch.optim.Adam(dynmodel.parameters() , lr = 1e-2)



loss_RK4 = torch.autograd.Variable(torch.tensor([0.],requires_grad=True))

###########################
###########################

running_loss_RK4 = 0.0

for iteration in range(5000):
    
    
    RK4_pred = rk4th_onestep_SparseId_non_rational(u[:-1],
                                               dynmodel,
                                               timestep,t= -1)
    
    loss_RK4 = torch.mean((u[1:] - RK4_pred) ** 2)
    
    loss_RK4.backward()

    optim.step()
    
    optim.zero_grad()
    
    running_loss_RK4 += loss_RK4.item()


print("**********************************")
print(dynmodel.linear.weight.detach().clone())

