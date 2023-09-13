#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:06:57 2022

@author: forootani
"""


import sys
import os
import time
import random

# from random import random,randrange
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
from Functions.utiles import initial_cond_generation
from Functions.linear_coeffs_module import CoeffsNetwork, CoeffsDictionaryRational

######################
from Functions.root_classes_non_rational import CompleteNetwork
from Functions.training_non_rational import train
from Functions.model_recovery_func import model_recovery
import Functions.plot_config_file

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(" We are using a " + device + " for the simulation")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


############################
############################
############################


def sensitivity_analysis(
    # estimated_coeffs_k,
    # library_k,
    param_set,
    RK_timestep,
    ts,
    list_initial_conditions=[[5, 2]],
):
    """
    Discription:
    the function that implement the sensitivity analysis for us,


    Args:
        param_set: a data class to set the input parameters for each dynamical system
        RK_timestep: time step that we use in the RK-sindy integration scheme
        ts: the time [t_min: step size :t_max]
        list_initial_conditions: a list containing the initial conditions

    Outputs: we just mention the most important output, the other outputs are out of 
            training loops

        coeff_noise_list: list of coeffs corresponding to each configuration
        and each algorithm.
    """

    #####################
    coeff_noise_list = []
    loss_values_list = []
    num_indp_var = len(list_initial_conditions[0])

    combinations_main = _combinations(
        param_set.num_indp_var,
        param_set.poly_order,
        include_interaction=True,
        include_bias=True,
        interaction_only=False,
    )

    coef_dim_row = sum(1 for _ in combinations_main)

    ####################

    for i in range(len(param_set.noise_level)):
        coeff_noise_neuron_list = []

        for l in range(len(param_set.num_hidden_neur)):
            set_all_seeds(42)

            for k in range(len(param_set.useRK)):
                for j in range(len(param_set.useNN)):
                    if param_set.useRK[k] == False and param_set.useNN[j] == False:
                        pass
                    else:
                        library_k = LibraryObject(
                            param_set.poly_order,
                            include_interaction=True,
                            include_bias=True,
                            interaction_only=False,
                            time_deriv_coef=2 / (param_set.t_max - param_set.t_min),
                        ).to(device)

                        estimated_coeffs_k = CoeffsNetwork(
                            coef_dim_row, param_set.num_indp_var, zero_inits=True
                        ).to(device)

                        network = Siren(
                            int(num_indp_var + 1),
                            [
                                param_set.num_hidden_neur[l],
                                param_set.num_hidden_neur[l],
                                param_set.num_hidden_neur[l],
                            ],
                            int(num_indp_var),
                        ).to(device)

                        model = (
                            CompleteNetwork(
                                network,
                                library_k,
                                estimated_coeffs_k,
                            )
                            .float()
                            .to(device)
                        )

                        # print("\n")
                        print("======================")
                        print(f"{model.estimated_coeffs_k.linear.weight}")
                        print("======================")

                        ############################
                        ############################
                        """ We can define our optimizer separately for
                            each dynamical system, e.g. for lorenz we use different setting
                            in param_set object we have an argument that we distinguish the 
                            names, ie. param_set.sys_model
                        """

                        optimizer = torch.optim.Adam(
                            [
                                {
                                    "params": model.func_approx.parameters(),
                                    "lr": 1e-4,
                                    "weight_decay": 0,
                                },
                                # {'params': model.estimated_coeffs.parameters(), 'lr': 1e-3, 'weight_decay': 0},
                                {
                                    "params": model.estimated_coeffs_k.parameters(),
                                    "lr": 1e-3,
                                    "weight_decay": 0,
                                },
                                #                {'params': model.library.parameters(), 'lr': 5e-3, 'weight_decay': 1e-4},
                            ]
                        )

                        if param_set.sys_model is "lorenz":
                            optimizer = torch.optim.Adam(
                                [
                                    {
                                        "params": model.func_approx.parameters(),
                                        "lr": 7e-4,
                                        "weight_decay": 0,
                                    },
                                    # {'params': model.estimated_coeffs.parameters(), 'lr': 1e-3, 'weight_decay': 0},
                                    {
                                        "params": model.estimated_coeffs_k.parameters(),
                                        "lr": 1e-2,
                                        "weight_decay": 0,
                                    },
                                    #                                {'params': model.library.parameters(), 'lr': 5e-3, 'weight_decay': 1e-4},
                                ]
                            )

                        #############################
                        #############################

                        print("\n")
                        y = Factory_Dyn_Sys(
                            ts,
                            list_initial_conditions,
                            param_set.fun_scaling_factor,
                            param_set.sys_model,
                            param_set.add_noise,
                            param_set.noise_level[i],
                        )

                        (
                            t_scaled_main,
                            initial_cond_main,
                            u_original_main,
                            list_t_scaled_main,
                            list_u_original,
                            list_initial_cond_main,
                            true_data_noise_free,
                        ) = y.run()

                        (
                            train_dataloader,
                            test_dataloader,
                        ) = train_test_spliting_dataset(
                            list_t_scaled_main,
                            list_initial_cond_main,
                            list_u_original,
                            device,
                            batch_size=2500,
                            split_value=0.9,
                            shuffle=False,
                        )

                        (
                            loss_values,
                            loss_values_NN,
                            loss_values_Coeff,
                            loss_values_RK4,
                            coeff_track_list,
                        ) = train(
                            model,
                            train_dataloader,
                            test_dataloader,
                            optimizer,
                            estimated_coeffs_k,
                            param_set.write_iterations,
                            param_set.max_iterations,
                            param_set.threshold_iteration,
                            param_set.threshold_value,
                            RK_timestep,
                            param_set.useRK[k],
                            param_set.useNN[j],
                        )
                        dic_coeff_neuron = {
                            "neuron": param_set.num_hidden_neur[l],
                            "useRK": param_set.useRK[k],
                            "useNN": param_set.useNN[j],
                            "learned_coeffs": model.estimated_coeffs_k.linear.weight.detach()
                            .clone()
                            .t()
                            .cpu()
                            .numpy(),
                        }
                        # self.linear.weight = torch.nn.Parameter(0 * self.linear.weight.clone().detach())

                        dic_coeff_noise_neuron = {
                            "noise": param_set.noise_level[i],
                            "neuron": param_set.num_hidden_neur[l],
                            "useRK": param_set.useRK[k],
                            "useNN": param_set.useNN[j],
                            "learned_coeffs": model.estimated_coeffs_k.linear.weight.detach()
                            .clone()
                            .t()
                            .cpu()
                            .numpy(),
                        }

                        dic_loss_values = {
                            "noise": param_set.noise_level[i],
                            "neuron": param_set.num_hidden_neur[l],
                            "useRK": param_set.useRK[k],
                            "useNN": param_set.useNN[j],
                            "loss_values": loss_values,
                            "loss_values_NN": loss_values_NN,
                            "loss_values_Coeff": loss_values_Coeff,
                        }

                        coeff_noise_neuron_list.append(dic_coeff_noise_neuron)

                        # coeff_noise_list.append(dic_coeff_noise)
                        loss_values_list.append(dic_loss_values)

                        model.estimated_coeffs_k.linear.weight = torch.nn.Parameter(
                            0 * model.estimated_coeffs_k.linear.weight.clone().detach()
                        )
                        # #model.reset_parameters()

                        """
                        Removing the entire architecture to avoid any memory confliction!
                        """

                        del (
                            model,
                            optimizer,
                            estimated_coeffs_k,
                            y,
                            train_dataloader,
                            test_dataloader,
                            network,
                            library_k,
                        )

                        torch.cuda.empty_cache()

        coeff_noise_list.append(coeff_noise_neuron_list)

    return (
        coeff_noise_list,
        loss_values_list,
        list_t_scaled_main,
        list_u_original,
        loss_values,
        loss_values_NN,
        loss_values_Coeff,
        loss_values_RK4,
        true_data_noise_free,
    )
