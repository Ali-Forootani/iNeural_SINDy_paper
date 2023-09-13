#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 08:51:43 2022

@author: forootani
"""

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
import matplotlib.pyplot as plt
import time


import sys, os
print( os.path.dirname( os.path.abspath('') ) )
sys.path.append( os.path.dirname( os.path.abspath('') ) )



from Functions.data_gen import (
 Interface_Dynamic_System, Factory_Dyn_Sys
 )

from Functions.nn_aprox_module import (NN_network_tanh, Siren,
                             NN_network_sigmoid, NN_network_relu, ODE_Net)

#from lib_first_order_equ import library_first_ord_eq,library_poly

from Functions.library import library_object,library_poly, _combinations, transform_torch
from Functions.data_set_preparation import data_set_fod, train_test_spliting_dataset
from torch.utils.data import DataLoader
from Functions.utiles import initial_cond_generation

import torch.nn as nn
#from Functions.train_module import train_func
from Functions.linear_coeffs_module import coeffs_network, coeffs_dictionary_rational

######################

#from Functions.sparse_estimators import Threshold
#from Functions.sparsity_scheduler import Periodic
#from Functions.constraint import LeastSquares

######################

from Functions.root_classes import complete_network
#from training import train
from Functions.training_rational import train
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


print(" We are using a "+ device + " for the simulation")

from random import random,randrange

import tikzplotlib
import Functions.plot_config_file


###################
###################



@dataclass
class Parameters_setting:
    t_max = 4
    t_min = 0
    num_samples = 1001
    num_init_cond = 3
    num_indp_var = 1
    min_init_cond = 2
    max_init_cond = 10
    poly_order = 1
    fun_scaling_factor = 0.1
    max_iterations = 5000
    write_iterations = 1000
    threshold_iteration = 2000
    threshold_value = 0.05
    save_model_path: str = './Results/First Order System/'
    tikz_save: bool = True
    add_noise: bool = True
    noise_level: float = 0.01
    shuffle = False
    useRK : bool = True
    useNN : bool = True
    tikz_save : bool = False





param_set = Parameters_setting()

os.makedirs(os.path.dirname(param_set.save_model_path), exist_ok=True)

ts = torch.linspace(param_set.t_min, param_set.t_max, param_set.num_samples)



RK_timestep = ts[1] - ts[0]

#print("here it is the time step")
#print(RK_timestep)


list_initial_conditions = initial_cond_generation(param_set.num_init_cond
                                                , param_set.num_indp_var
                                                , param_set.min_init_cond, param_set.max_init_cond)

#######################
#######################



# First Order Dynamic System

y = Factory_Dyn_Sys(ts,list_initial_conditions,
                    param_set.fun_scaling_factor,
                    "First Order Dynamic System",param_set.add_noise, param_set.noise_level)



(t_scaled_main, initial_cond_main,
        u_original_main,
        list_t_scaled_main, list_u_original, list_initial_cond_main) = y.run()



########################




####################
#################### Plotting the curves

#param_set.num_indp_var

for i in range(int(param_set.num_init_cond)):
    for j in range(int(param_set.num_indp_var)):
        
        plt.plot(list_t_scaled_main[i], 1/(param_set.fun_scaling_factor) * list_u_original[i][:,j],linewidth= 3)
 
plt.xlabel('t')
plt.ylabel('X(t)')
plt.title("Different initial conditions")

if param_set.tikz_save:
    tikzplotlib.save( param_set.save_model_path + "Different initial conditions.tex")

plt.savefig(param_set.save_model_path + "Different initial conditions.png")
    
plt.show()
####################


train_dataloader, test_dataloader = train_test_spliting_dataset(list_t_scaled_main,
                                                                list_initial_cond_main,
                                list_u_original, device, batch_size=2000, split_value=0.9, shuffle=False)


network = Siren(int(param_set.num_indp_var+1),[32, 32, 32], int(param_set.num_indp_var))


loss_fn = nn.MSELoss()

####################
####################
####################


time_deriv_coef = 2/(param_set.t_max - param_set.t_min)


library = library_object(param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False,
                               time_deriv_coef=2/(param_set.t_max - param_set.t_min))


library_k = library_object(param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False,
                               time_deriv_coef=2/(param_set.t_max - param_set.t_min))


##############################


combinations_main =_combinations(param_set.num_indp_var, param_set.poly_order, include_interaction=True,
                               include_bias=True,
                               interaction_only=False)

coef_dim_row = sum(1 for _ in combinations_main)


estimated_coeffs = coeffs_dictionary_rational(coef_dim_row , param_set.num_indp_var)

estimated_coeffs_k = coeffs_network(coef_dim_row , param_set.num_indp_var)



#####################
#####################
#####################




model = complete_network( network, library, library_k,
                         #estimator, constraint,
                         estimated_coeffs, estimated_coeffs_k,
                         ).to(device)


model = model.float()


optimizer = torch.optim.Adam([
                {'params': model.func_approx.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
                {'params': model.estimated_coeffs.parameters(), 'lr': 1e-2, 'weight_decay': 1e-4},
                {'params': model.estimated_coeffs_k.parameters(), 'lr': 1e-2, 'weight_decay': 0},
#                {'params': model.library.parameters(), 'lr': 5e-3, 'weight_decay': 1e-4},
            ])




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

func_arguments.setdefault("")


######################################
######################################

""" 
Calling the training function and identifying the model
"""


start = time.time()
                                        
loss_values, loss_values_NN, loss_values_Coeff, loss_values_RK4 = train(func_arguments)

end = time.time()


'''
To go from np.array to cpu Tensor, use torch.from_numpy().
To go from cpu Tensor to gpu Tensor, use .cuda().
To go from a Tensor that requires_grad to one that does not, use .detach() (in your case, your net output will most likely requires gradients and so it’s output will need to be detached).
To go from a gpu Tensor to cpu Tensor, use .cpu().
To go from a cpu Tensor to np.array, use .numpy().
'''

print("\n")
print(f"The elapsed time is: {end-start} " + "seconds \n")
print(f"The ODE coefficients by our approach are:\n {model.estimated_coeffs_k.linear.weight.detach().clone().t().cpu().numpy()}" )



Learned_Coeffs = model.estimated_coeffs_k.linear.weight.detach().clone().t().cpu().numpy()

######################################
######################################


""" 
Recovering the learned model and comparing it with the original model
"""

f = lambda z: (transform_torch( torch.tensor(z).reshape(1,-1), param_set.poly_order,
                               True, True, False)@Learned_Coeffs).numpy()
learnt_deri = lambda z,t: f(z)

x0 = np.array(list_initial_conditions[0])
ts_refine = np.arange(0,param_set.t_max,1e-2)
sol_learnt = solve_ivp(lambda t, x: learnt_deri(x, t), 
            [ts_refine[0], ts_refine[-1]], x0, t_eval=ts_refine)
x_learnt = np.transpose(sol_learnt.y).reshape(1,-1,param_set.num_indp_var)

#####################################
######################################


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
    tikzplotlib.save( param_set.save_model_path + "first_order_system_training_loss.tex")
plt.savefig(param_set.save_model_path + "first_order_training_loss.png")









fig_4 = plt.figure(4)
plt.plot(ts, (1/param_set.fun_scaling_factor) * list_u_original[0][:,0], label=r"$Model$",linestyle='-',linewidth=6,color='r')
plt.plot(ts_refine, x_learnt[0,:, 0], label=r"$DeepSindy-RK4$",linestyle='-',linewidth=2,color='k')
plt.xlabel('t')
plt.ylabel('X(t)')
plt.legend()

if param_set.tikz_save:
    tikzplotlib.save( param_set.save_model_path + "first_order_system.tex")
plt.savefig(param_set.save_model_path + "first_order_system.pdf")
plt.show()


