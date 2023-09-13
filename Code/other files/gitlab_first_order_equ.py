#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:48:18 2022

@author: forootani
"""


import numpy as np
import torch
from scipy.integrate import odeint, ode, solve_ivp
from dataclasses import dataclass



from data_gen import (first_order_solution,
 exp_function, rational, rational_2D, rational_nD, rational_7D,
 
 Interface_Dynamic_System, Factory_Dyn_Sys
 )

import matplotlib.pyplot as plt

from nn_aprox_module import (NN_network_tanh, Siren,
                             NN_network_sigmoid, NN_network_relu, ODE_Net)

#from lib_first_order_equ import library_first_ord_eq,library_poly


from lib_rational import library_first_ord_eq,library_poly, _combinations


from data_set_preparation import data_set_fod, train_test_spliting_dataset


from torch.utils.data import DataLoader
from utiles import initial_cond_generation

import torch.nn as nn
from train_module import train_func

from linear_coeffs_module import coeffs_network, coeffs_dictionary_rational


######################

from sparse_estimators import Threshold
from sparsity_scheduler import Periodic
from constraint import LeastSquares

######################



from root_classes import complete_network



#from training import train

from training_rational import train_2


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(device)



from random import random,randrange

###################
###################


#####################
#####################
### Making a list of initial conditions

#list_initial_conditions=[]
#element_init=[]



#@dataclass
#class Parameters_setting:


@dataclass
class Parameters_setting:
    t_max = 0.3
    t_min = 0
    num_samples = 201
    num_init_cond = 5
    num_indp_var = 1
    min_init_cond = 1
    max_init_cond = 5
    poly_order = 1
    fun_scaling_factor = 0.1
    max_iterations = 30000
    write_iterations = 5000
    threshold_iteration = 3000
    threshold_value = 0.01
    shuffle=False




param_set = Parameters_setting()

ts = torch.linspace(param_set.t_min, param_set.t_max, param_set.num_samples)



RK_timestep = ts[1] - ts[0]

print("here it is the time step")
print(RK_timestep)


list_initial_conditions=initial_cond_generation(param_set.num_init_cond
                                                , param_set.num_indp_var
                                                , param_set.min_init_cond, param_set.max_init_cond)





#######################
#######################



#Factory_Dyn_Sys.function_data(ts, list_initial_conditions, fun_scaling_factor,"lorenz")


#list_initial_conditions = [[-8,7,27],[-6,6,25],[-9,8,22]]


# First Order Dynamic System

y = Factory_Dyn_Sys(ts,list_initial_conditions,param_set.fun_scaling_factor,"First Order Dynamic System")



#t_scaled_main, initial_cond_main, u_original_main, u_original = y.run()

(t_scaled_main, initial_cond_main,
        u_original_main,
        list_t_scaled_main, list_u_original, list_initial_cond_main) = y.run()


########################






####################
#################### Plotting the curves

#param_set.num_indp_var

for i in range(int(param_set.num_init_cond * 1)):
    for j in range(int(param_set.num_indp_var)):
        
        plt.plot(list_t_scaled_main[i], list_u_original[i][:,j],linewidth= 3)
   
   
    #im = ax.scatter(list_t_scaled_main[0], list_u_original[i][:,1], marker="+", s=10)
plt.show()
####################




train_dataloader, test_dataloader = train_test_spliting_dataset(list_t_scaled_main,
                                                                list_initial_cond_main,
                                list_u_original, device, batch_size=2500, split_value=0.9, shuffle=False)




###################network_2 = Siren(2, [8, 8], 1)


#network_2 = Siren(3, [32, 32, 32], 2)
##### (- 1 * x)/(1 + x), (-y)/(1 + x + x * x), 'lr': 1e-4, lr: 1e-3, batches: 70000, iteration threshold: 15000 , threshold: 0.2

#network_2 = Siren(4, [16, 16, 16, 16], 3) 
##works for three variables

#network_2 = Siren(4, [32, 32, 32], 3) 
#### 'lr': 1e-4, lr: 1e-3, batches: 70000, iteration threshold: 15000 , threshold: 0.2


network_2 = Siren(int(param_set.num_indp_var+1),[32, 32], int(param_set.num_indp_var))
### 'lr': 1e-4, lr: 1e-3, batches: 70000, iteration threshold: 15000, threshold: 0.2
### (- 1 * x)/(1 +  x + x * y ), (-y -x)/(1 + y + x * x * x * x * x)


#network_2 = Siren(3, [32, 32, 32], 2) works for two variables, lr': 4e-4, lr': 1e-3


#network_3 = NN_network_sigmoid(1, [50, 50, 50, 50], 1)


#network_4 = NN_network_tanh(2, [8, 8], 1)



loss_fn = nn.MSELoss()


#n_features = axis.shape[1]

#n_features = 6

####################
####################
####################


time_deriv_coef = 2/(param_set.t_max - param_set.t_min)


library = library_first_ord_eq(param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False,
                               time_deriv_coef=2/(param_set.t_max - param_set.t_min))


library_k = library_first_ord_eq(param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False,
                               time_deriv_coef=2/(param_set.t_max - param_set.t_min))


##############################

#num_resblks_ODE = 4
#hidden_features_ODE = 20
#print_models = True 



#ODE_Net(
#    n=2,
#    num_residual_blocks= num_resblks_ODE,
#    hidden_features= hidden_features_ODE,
#    print_model=print_models,
#)



combinations_main =_combinations(param_set.num_indp_var, param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False)

coef_dim_row = sum(1 for _ in combinations_main)


estimated_coeffs = coeffs_dictionary_rational(coef_dim_row , param_set.num_indp_var)

estimated_coeffs_k = coeffs_network(coef_dim_row , param_set.num_indp_var)




#####################
#####################
#####################






model = complete_network( network_2, library, library_k,
                         #estimator, constraint,
                         estimated_coeffs, estimated_coeffs_k,
                         ).to(device)


model = model.float()



optimizer = torch.optim.Adam([
                {'params': model.func_approx.parameters(), 'lr': 1e-5, 'weight_decay': 0},
                {'params': model.estimated_coeffs.parameters(), 'lr': 1e-3, 'weight_decay': 0},
                {'params': model.estimated_coeffs_k.parameters(), 'lr': 1e-3, 'weight_decay': 0},
#                {'params': model.library.parameters(), 'lr': 5e-3, 'weight_decay': 1e-4},
            ])




unshuffled_data = torch.cat((t_scaled_main, initial_cond_main),1)


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
                  "useOnlyRK" : False,
                  "useOnlyNN" : False
                  }



loss_values, loss_values_NN, loss_values_Coeff, loss_values_RK4 = train_2(func_arguments)


print("=============")



print("The ODE coefficients by our approach are:")

#print(model.constraint_coeffs(scaled=False, sparse=True))


#print(model.estimated_coeffs.numerator.weight.detach().clone().t().numpy())
#print("*****************")
#print(model.estimated_coeffs.denominator.weight.detach().clone().t().numpy())
print("*****************")
print(model.estimated_coeffs_k.linear.weight.detach().clone().t().numpy())


#rational_object_2.simplification()

print("multiply the coeffiecint by 2/(t_max -t_min) to get the true value")


plt.subplot(1, 4, 1)
plt.semilogy(loss_values)
plt.subplot(1, 4, 2)
plt.semilogy(loss_values_NN)
plt.subplot(1, 4, 3)
plt.semilogy(loss_values_Coeff)
plt.subplot(1, 4, 4)
plt.semilogy(loss_values_RK4)
