#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:48:18 2022

@author: forootani
"""



###################
###################


import sys
import os
import time
from random import random,randrange
from scipy.integrate import odeint, ode, solve_ivp
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
import tikzplotlib
import torch.nn as nn
print( os.path.dirname( os.path.abspath('') ) )
sys.path.append( os.path.dirname( os.path.abspath('') ) )
from Functions.data_gen import (Interface_Dynamic_System, Factory_Dyn_Sys)
from Functions.nn_aprox_module import (Siren, Costum_NN, ODENet)
from Functions.library import LibraryObject, library_poly, _combinations, transform_torch
from Functions.data_set_preparation import DataSetFod, train_test_spliting_dataset
from torch.utils.data import DataLoader
from Functions.utiles import initial_cond_generation
from Functions.linear_coeffs_module import CoeffsNetwork, CoeffsDictionaryRational
######################
from Functions.root_classes_non_rational import CompleteNetwork
from Functions.training_non_rational import train
from Functions.model_recovery_func import model_recovery, model_recovery_single_noise_level
import Functions.plot_config_file

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(" We are using a "+ device + " for the simulation")






#####################
#####################
### Making a list of initial conditions

#list_initial_conditions=[]
#element_init=[]



#@dataclass
#class Parameters_setting:


@dataclass
class Parameters_setting:
    t_max = 20
    t_min = 0
    num_samples = 201
    num_init_cond = 3
    num_indp_var = 3
    min_init_cond = 1
    max_init_cond = 3
    poly_order = 2
    fun_scaling_factor = 1
    max_iterations = 50000
    write_iterations = 5000
    threshold_iteration = 30000
    threshold_value = 0.02
    shuffle=False
    save_model_path: str = './Results/3_D_Non_Linear/'
    add_noise: bool = True
    noise_level: float = 0.05
    shuffle = False
    tikz_save : bool = False
    useRK : bool = True
    useNN : bool = True
    
    
    





"Creating a dataclass object and time step "

 
param_set = Parameters_setting()
ts = torch.linspace(param_set.t_min, param_set.t_max, param_set.num_samples)

os.makedirs(os.path.dirname(param_set.save_model_path), exist_ok=True)


RK_timestep = ts[1] - ts[0]
print("here it is the time step:" , float(RK_timestep))




######################
"Creating initial conditions for a given range"
list_initial_conditions=initial_cond_generation(param_set.num_init_cond
                                                , param_set.num_indp_var
                                                , param_set.min_init_cond, param_set.max_init_cond)




#######################
#######################

"""Making an instance for a desired dynamic system and creating the data set
by typing the name of the system, e.g. lorenz, Fitz-Hugh Nagumo,
Cubic Damped Oscillator, etc.
"""

y = Factory_Dyn_Sys(ts,list_initial_conditions,
                    param_set.fun_scaling_factor,
                    "3_D_Non_Linear",
                    param_set.add_noise,
                    param_set.noise_level)


(t_scaled_main, initial_cond_main,
        u_original_main,
        list_t_scaled_main,
        list_u_original,
        list_initial_cond_main,
        true_data_noise_free) = y.run()


#######################
####################### 

"""Visualizing the curves for different initial conditions"""

#param_set.num_indp_var

for i in range(int(param_set.num_init_cond)):
    for j in range(int(param_set.num_indp_var)):
        
        plt.plot(list_t_scaled_main[i], list_u_original[i][:,j],linewidth= 3)   

plt.xlabel('t')
plt.ylabel('X(t)')
plt.title("Different initial conditions")    

plt.show()
####################

"""Creating data loader"""


train_dataloader, test_dataloader = train_test_spliting_dataset(list_t_scaled_main,
                                                                list_initial_cond_main,
                                list_u_original, device, batch_size=2500, split_value=0.9, shuffle=False)

"""Defining Neural Network framework with desired activation function"""

network = Siren(int(param_set.num_indp_var+1),[32, 32, 32], int(param_set.num_indp_var))



####################
####################
####################

"""
library: instance of the polynomial library to make use in our algorithm
time_deriv_coef: here it is the concept, dx/dt = time_deriv_coef
"""


time_deriv_coef = 2/(param_set.t_max - param_set.t_min)


#library = LibraryObject(param_set.poly_order, include_interaction=True,
#                               include_bias=True,
#                               interaction_only=False,
#                               time_deriv_coef=2/(param_set.t_max - param_set.t_min))


library_k = LibraryObject(param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False,
                               time_deriv_coef=2/(param_set.t_max - param_set.t_min))


##############################



combinations_main =_combinations(param_set.num_indp_var, param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False)

coef_dim_row = sum(1 for _ in combinations_main)


#estimated_coeffs = coeffs_dictionary_rational(coef_dim_row , param_set.num_indp_var)

estimated_coeffs_k = CoeffsNetwork(coef_dim_row , param_set.num_indp_var)




#####################
#####################
#####################



model = CompleteNetwork(network,
                        #library,
                        library_k,
                        #estimated_coeffs,
                         estimated_coeffs_k,
                         ).float().to(device)



optimizer = torch.optim.Adam([
                {'params': model.func_approx.parameters(), 'lr': 1e-4, 'weight_decay': 0},
                #{'params': model.estimated_coeffs.parameters(), 'lr': 1e-3, 'weight_decay': 0},
                {'params': model.estimated_coeffs_k.parameters(), 'lr': 1e-3, 'weight_decay': 0},
            ])



(loss_values, loss_values_NN, 
 loss_values_Coeff, loss_values_RK4
 , coeff_track_list) = train(model,
                                        train_dataloader,
                                        test_dataloader,
                                        optimizer,
                                        estimated_coeffs_k,
                                        param_set.write_iterations,
                                        param_set.max_iterations,
                                        param_set.threshold_iteration,
                                        param_set.threshold_value,
                                        RK_timestep,
                                        param_set.useRK,
                                        param_set.useNN 
                                        )























"""
func_arguments = {"model" : model,
                  "train_dataloader" : train_dataloader,
                  "test_dataloader" : test_dataloader,
                  "optimizer" : optimizer,
                  "estimated_coeffs" : estimated_coeffs,
                  "estimated_coeffs_k" : estimated_coeffs_k,
                  "write_iterations" : param_set.write_iterations,
                  "max_iterations" : param_set.max_iterations,
                  "threshold_iteration" : param_set.threshold_iteration,
                  "threshold_value" : param_set.threshold_value,
                  "RK_timestep" : RK_timestep,
                  "useRK" : param_set.useRK,
                  "useNN" : param_set.useNN
                  }
loss_values, loss_values_NN, loss_values_Coeff, loss_values_RK4 = train(func_arguments)
"""
###########################
###########################




'''
To go from np.array to cpu Tensor, use torch.from_numpy().
To go from cpu Tensor to gpu Tensor, use .cuda().
To go from a Tensor that requires_grad to one that does not, use .detach() (in your case, your net output will most likely requires gradients and so it’s output will need to be detached).
To go from a gpu Tensor to cpu Tensor, use .cpu().
To go from a cpu Tensor to np.array, use .numpy().
'''

print("\n")

print(f"The ODE coefficients by our approach are:\n {model.estimated_coeffs_k.linear.weight.detach().clone().t().cpu().numpy()}" )

Learned_Coeffs = model.estimated_coeffs_k.linear.weight.detach().clone().t().cpu().numpy()







####################################
####################################




f = lambda z: (transform_torch(torch.tensor(z).reshape(1,-1), param_set.poly_order,
                               True, True, False
                               )@Learned_Coeffs)

learnt_deri = lambda z,t: f(z)



x0 = np.array(list_initial_conditions[0])
ts_refine = np.arange(0,param_set.t_max,1e-2)


sol_learnt = solve_ivp(lambda t, x: learnt_deri(x, t), 
            [ts_refine[0], ts_refine[-1]], x0, t_eval=ts_refine)


x_learnt = np.transpose(sol_learnt.y).reshape(1,-1,param_set.num_indp_var)



##############################
##############################



""" Plotting the results"""

plt.figure(1)
plt.subplot(1, 4, 1)
plt.semilogy(loss_values)
plt.grid(True, which ="both")
plt.xlabel('Total loss',fontsize=14)


plt.subplots_adjust(left=0.15)
plt.subplot(1, 4, 2)
plt.semilogy(loss_values_NN)
plt.grid(True, which ="both")
plt.xlabel('NN loss',fontsize=14)


plt.subplots_adjust(left=0.15)
plt.subplot(1, 4, 3)
plt.semilogy(loss_values_Coeff)
plt.grid(True, which ="both")
plt.xlabel('Coefficient loss',fontsize=14)

plt.subplots_adjust(left=0.15)
plt.subplot(1, 4, 4)
plt.semilogy(loss_values_RK4)
plt.grid(True, which ="both")
#plt.xticks(fontname="serif" ,fontsize=8)
plt.xlabel('Rk4 loss',fontsize=14)
plt.subplots_adjust(left=0.15)


if param_set.tikz_save:
    tikzplotlib.save( param_set.save_model_path + "3_D_None_Linear.tex")
plt.savefig(param_set.save_model_path + ".png")


#############################
#############################




fig_2 = plt.figure(3)

ax = fig_2.add_subplot(121, projection="3d")
ax.plot(
    x_learnt[0,:, 0],
    x_learnt[0,:, 1],
    x_learnt[0,:, 2], "r", linewidth=4, label= "DeepSindy-RK4"
)

ax.plot(
    1/(param_set.fun_scaling_factor) * list_u_original[0][:,0],
    1/(param_set.fun_scaling_factor) * list_u_original[0][:,1],
    1/(param_set.fun_scaling_factor) * list_u_original[0][:,2],
    'k-', linewidth = 1, label= "Model"
)
ax.legend()
ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$",)



ax = fig_2.add_subplot(122, projection="3d")
ax.plot(
    1/(param_set.fun_scaling_factor) * list_u_original[0][:,0],
    1/(param_set.fun_scaling_factor) * list_u_original[0][:,1],
    1/(param_set.fun_scaling_factor) * list_u_original[0][:,2],
    'k-',
)
#plt.title("Model")
ax.set(xlabel= "$x$", ylabel= "$y$", zlabel= "$z$", title="Model")
plt.savefig(param_set.save_model_path + "3_D_Non_Linear.png")





