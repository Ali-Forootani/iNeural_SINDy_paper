#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:38:18 2022

@author: forootani
"""

import sys
import os
import time
import pickle
from random import random, randrange
from scipy.integrate import odeint, ode, solve_ivp
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
import tikzplotlib
import torch.nn as nn

print(os.path.dirname(os.path.abspath("")))
sys.path.append(os.path.dirname(os.path.abspath("")))
from Functions.data_gen import Interface_Dynamic_System, Factory_Dyn_Sys
from Functions.nn_aprox_module import Siren, Costum_NN, ODENet
from Functions.library import (
    LibraryObject,
    library_poly,
    _combinations,
    transform_torch,
)
from Functions.data_set_preparation import DataSetFod, train_test_spliting_dataset
from torch.utils.data import DataLoader
from Functions.utiles import initial_cond_generation, HeatmapSetting
from Functions.linear_coeffs_module import CoeffsNetwork, CoeffsDictionaryRational

######################
from Functions.root_classes_non_rational import CompleteNetwork
from Functions.training_non_rational import train
from Functions.model_recovery_func import (
    model_recovery,
    post_proscesing_2,
    model_recovery_single_noise_level,
)
from Functions.simulation_sensitivity import sensitivity_analysis
import Functions.plot_config_file

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("We are using a__" + device + "__for the simulation")
#####################
#####################


@dataclass
class ParametersSetting:
    """
    Discription:
    a data class to set the input parameters

    Args:
        noise_level: A list of noise values can be given, e.g. [0, 0.02, 0.04]
                     The curves are plotted only for case of single noise value, e.g. [0.08] or [0.04]
                     If you increase the size of the noise level list, then modify the plotting section
        useRK: boolian, to activate RK4-SINDy method
        useNN: boolian, to activate iNeuralSINDy method
                        to compare all the methods we should consider [True,True],[True,False], [False, True]
    """

    t_max = 200
    t_min = 0
    num_samples = [151, 201, 251, 301, 351, 401, 451]
    num_init_cond = 1
    num_indp_var = 2
    min_init_cond = 1
    max_init_cond = 4
    poly_order = 3
    num_hidden_neur = [32]
    fun_scaling_factor = 1
    max_iterations = 50000
    write_iterations = 5000
    threshold_iteration = 15000
    threshold_value = 0.05
    save_model_path: str = "./Results/Fitz_Hugh_Nagumo/"
    add_noise: bool = True
    noise_level = list(np.arange(0.00, 0.08, 0.02))
    shuffle = False
    useRK = [True, False]
    useNN = [True, False]
    tikz_save: bool = False
    name_pic_loss: str = "Fitz_Hugh_Nagumo_loss_noise_"
    name_pic_data: str = "Fitz_Hugh_Nagumo_training_data_noise_"
    name_pic_comparison: str = "Fitz_Hugh_Nagumo_noise_"
    sys_model: str = "Fitz-Hugh Nagumo"

########################
param_set = ParametersSetting()

os.makedirs(os.path.dirname(param_set.save_model_path), exist_ok=True)

###################
###################
###################

"""
time_deriv_coef: here it is the concept, dx/dt = time_deriv_coef * \theta * coeffs 
"""

time_deriv_coef = 2 / (param_set.t_max - param_set.t_min)


start = time.time()

###################################
###################################
###################################
###################################

"""Creating list of initial conditions for a given range or inserting manually 
    the initial conditions
"""

list_initial_conditions = initial_cond_generation(
    param_set.num_init_cond,
    param_set.num_indp_var,
    param_set.min_init_cond,
    param_set.max_init_cond,
)

#######################
#######################

""" Some useful torch syntaxes
To go from np.array to cpu Tensor, use torch.from_numpy().
To go from cpu Tensor to gpu Tensor, use .cuda().
To go from a Tensor that requires_grad to one that does not, use .detach() (in your case, your net output will most likely requires gradients and so it’s output will need to be detached).
To go from a gpu Tensor to cpu Tensor, use .cpu().
To go from a cpu Tensor to np.array, use .numpy().
"""


coeff_noise_list = []
loss_values_list = []



list_initial_conditions = [[3, 2]]
####################


""" Main simulation loop """

er_curves_samples_NNRK_list = []
er_curves_samples_NN_list = []


for i in range(len(param_set.num_samples)):
   

    ts = torch.linspace(param_set.t_min, param_set.t_max, param_set.num_samples[i])

    RK_timestep = ts[1] - ts[0]

    (
        coeff_noise_list,
        loss_values_list,
        list_t_scaled_main,
        list_u_original,
        loss_values,
        loss_values_NN,
        loss_values_Coeff,
        loss_values_RK4,
        true_data_noise_free,
    ) = sensitivity_analysis(
        param_set,
        RK_timestep,
        ts,
        list_initial_conditions=[[3, 2]],
    )

    (
        er_coef_NNRK_list,
        er_coef_RK_list,
        er_coef_NN_list,
        er_curves_NNRK_list,
        er_curves_RK_list,
        er_curves_NN_list,
    ) = post_proscesing_2(
        coeff_noise_list, param_set, NN_sensitivity=False, Sample_sensitivity=True
    )

    er_curves_NNRK_list.append(param_set.num_samples[i])
    er_curves_samples_NNRK_list.append(er_curves_NNRK_list)

    er_curves_NN_list.append(param_set.num_samples[i])
    er_curves_samples_NN_list.append(er_curves_NN_list)

###########################################
###########################################
###########################################


"""Saving the sensitivity analysis in the directory with the same name """



with open(f"{param_set.save_model_path}" +"coeff_noise_list_diff_samples", "wb") as fp:
    pickle.dump(coeff_noise_list, fp)
    
with open(f"{param_set.save_model_path}" + "er_curves_samples_NNRK_list_diff_samples", "wb") as fp:
    pickle.dump(er_curves_samples_NNRK_list, fp)
    
with open(f"{param_set.save_model_path}" + "er_curves_samples_NN_list_diff_samples", "wb") as fp:
    pickle.dump(er_curves_samples_NN_list, fp)



with open(f"{param_set.save_model_path}" +"coeff_noise_list_diff_samples", "rb") as fp:
    coeff_noise_list = pickle.load(fp)

with open(f"{param_set.save_model_path}" +"er_curves_samples_NNRK_list_diff_samples", "rb") as fp:
    er_curves_samples_NNRK_list = pickle.load(fp)

with open(f"{param_set.save_model_path}" +"er_curves_samples_NN_list_diff_samples", "rb") as fp:
    er_curves_samples_NN_list = pickle.load(fp)




###########################################
###########################################
###########################################

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


#####################
#####################

x_1_er_sens_array_NNRK = np.zeros(
    (len(param_set.noise_level), len(param_set.num_samples))
)
x_2_er_sens_array_NNRK = np.zeros(
    (len(param_set.noise_level), len(param_set.num_samples))
)


x_1_er_sens_array_NN = np.zeros(
    (len(param_set.noise_level), len(param_set.num_samples))
)
x_2_er_sens_array_NN = np.zeros(
    (len(param_set.noise_level), len(param_set.num_samples))
)


for l in range(len(param_set.noise_level)):
    for i in range(len(param_set.num_samples)):
        x_1_er_sens_array_NNRK[l, i] = er_curves_samples_NNRK_list[i][l][0][1]
        x_2_er_sens_array_NNRK[l, i] = er_curves_samples_NNRK_list[i][l][0][2]

        x_1_er_sens_array_NN[l, i] = er_curves_samples_NN_list[i][l][0][1]
        x_2_er_sens_array_NN[l, i] = er_curves_samples_NN_list[i][l][0][2]


#######################
#######################
#######################


""" 
making an instance for heatmap plot setting
"""

param_heatmap = HeatmapSetting(v_max = 3, fontsize = 12)
kwargs = param_heatmap()




#######################
#######################
#######################


import seaborn as sns

plt.rcParams.update(
    {
        "font.size": 15,
    }
)

fig_2 = plt.figure(figsize=(10, 6.2))

font_size = {"fontsize": 12}

# create 2x1 subfigs
subfigs = fig_2.subfigures(nrows=2, ncols=1)

# create an axis for the color bar
cbar_ax = fig_2.add_axes([0.99, 0.3, 0.02, 0.4])


for row, subfig in enumerate(subfigs):
    if row == 0:
        subfig.suptitle(r"\texttt{iNeural-SINDy}")
        axs = subfig.subplots(nrows=1, ncols=2)

        g = sns.heatmap(
            x_1_er_sens_array_NNRK,
            xticklabels=np.array(param_set.num_samples) - 1,
            yticklabels=param_set.noise_level,
            ax=axs[0],
            cbar=False,
            **kwargs,
        )

        axs[0].set_title(r"$\mathcal{E}_{\mathbf{x}_1}$")
        axs[0].set_ylabel(
            "Noise level",
        )
        axs[0].set_xlabel(
            "Number of samples",
        )

        g = sns.heatmap(
            x_2_er_sens_array_NNRK,
            xticklabels=np.array(param_set.num_samples) - 1,
            yticklabels=param_set.noise_level,
            ax=axs[1],
            cbar=False,
            **kwargs,
        )
        axs[1].set_title(r"$\mathcal{E}_{\mathbf{x}_2}$")
        axs[1].set_xlabel(
            "Number of samples",
        )

    if row == 1:
        subfig.suptitle(r"\texttt{DeePyMoD}")
        axs = subfig.subplots(nrows=1, ncols=2)

        g = sns.heatmap(
            x_1_er_sens_array_NN,
            xticklabels=np.array(param_set.num_samples) - 1,
            yticklabels=param_set.noise_level,
            ax=axs[0],
            cbar=False,
            **kwargs,
        )

        axs[0].set_title(r"$\mathcal{E}_{\mathbf{x}_1}$")
        axs[0].set_ylabel(
            "Noise level",
        )
        axs[0].set_xlabel(
            "Number of samples",
        )

        g = sns.heatmap(
            x_2_er_sens_array_NN,
            xticklabels=np.array(param_set.num_samples) - 1,
            yticklabels=param_set.noise_level,
            ax=axs[1],
            cbar_ax=cbar_ax,
            cbar=True,
            **kwargs
        )
        axs[1].set_title(r"$\mathcal{E}_{\mathbf{x}_2}$")
        axs[1].set_xlabel(
            "Number of samples",
        )

# plt.tight_layout()


plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_comparison
    + "_heatmap_"
    + "_different_samples_"
    + "_threshold_"
    + str(param_set.threshold_value)
    + "iNeuralSINDy"
    + ".png",
    bbox_inches="tight",
    dpi=300,
)
plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_comparison
    + "_heatmap_"
    + "_different_samples_"
    + "_threshold_"
    + str(param_set.threshold_value)
    + "iNeuralSINDy"
    + ".pdf",
    bbox_inches="tight",
    dpi=300,
)


#######################
#######################
#######################


plt.show()
####################
end = time.time()

