#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:15:12 2022

@author: forootani
"""

from dataclasses import dataclass
import numpy as np
import torch
import pickle
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Functions.library import (
    #    LibraryObject,
    #    library_poly,
    #    _combinations,
    transform_torch,
)

# from Functions.data_set_preparation import DataSetFod, train_test_spliting_dataset
######################################
######################################


def model_recovery_single_noise_level(
    coeff_noise_list, list_initial_conditions, t_max, poly_order, num_indp_var, t_step
):
    """
    Discription:
                This function recovers the estimated model by soliving numerical integration
    Args:
        coeff_noise_list: a list containing the estimated coefficients with RK4, iNeuralSINDy, and NeuralSINDy
        list_initial_conditions: a list containing all the initial conditions
        t_max: maximum time to simulate and do the integration
        poly_order: order of ploynomial that we consider to construct the library
        num_indp_var: number of independent variable, e.g. x,y,z ...
        t_step: time step that is choosen to make use in this term np.arange(0, t_max, t_step)
    Output:
        x_learnt_list: a list containing the solution of estimated model by numerical integration
    """
    x_learnt_list = []
    for i in range(len(coeff_noise_list)):
        dic_element = coeff_noise_list[i].copy()
        for key in coeff_noise_list[i]:
            if key == "learned_coeffs":
                Learned_Coeffs = coeff_noise_list[i][key]
                f = lambda z: (
                    transform_torch(
                        torch.tensor(z).reshape(1, -1), poly_order, True, True, False
                    )
                    @ Learned_Coeffs
                )
                learnt_deri = lambda z, t: f(z)
                x0 = np.array(list_initial_conditions[0])
                ts_refine = np.arange(0, t_max, t_step)
                sol_learnt = solve_ivp(
                    lambda t, x: learnt_deri(x, t),
                    [ts_refine[0], ts_refine[-1]],
                    x0,
                    t_eval=ts_refine,
                )
                x_learnt = np.transpose(sol_learnt.y).reshape(1, -1, num_indp_var)
                dic_element["x_learnt"] = x_learnt
                x_learnt_list.append(dic_element)
    return x_learnt_list


def model_recovery(
    coeff_noise_list, list_initial_conditions, t_max, poly_order, num_indp_var, t_step
):
    """
    Discription:
                This function recovers the estimated model by soliving numerical integration
                This function is more general compared with model_recovery_single_noise_level
    Args:
        coeff_noise_list: a list containing the estimated coefficients with RK4, iNeuralSINDy, and NeuralSINDy
        list_initial_conditions: a list containing all the initial conditions
        t_max: maximum time to simulate and do the integration
        poly_order: order of ploynomial that we consider to construct the library
        num_indp_var: number of independent variable, e.g. x,y,z ...
        t_step: time step that is choosen to make use in this term np.arange(0, t_max, t_step)
    Output:
        x_learnt_list: a list containing the solution of estimated model by numerical integration
    """
    x_learnt_noise_neuron_list = []

    for l in range(len(coeff_noise_list)):
        x_learnt_list = []
        for i in range(len(coeff_noise_list[l])):
            dic_element = coeff_noise_list[l][i].copy()

            for key in coeff_noise_list[l][i]:
                if key == "learned_coeffs":
                    Learned_Coeffs = coeff_noise_list[l][i][key]
                    f = lambda z: (
                        transform_torch(
                            torch.tensor(z).reshape(1, -1),
                            poly_order,
                            True,
                            True,
                            False,
                        )
                        @ Learned_Coeffs
                    )
                    learnt_deri = lambda z, t: f(z)
                    x0 = np.array(list_initial_conditions[0])
                    ts_refine = np.arange(0, t_max, t_step)
                    sol_learnt = solve_ivp(
                        lambda t, x: learnt_deri(x, t),
                        [ts_refine[0], ts_refine[-1]],
                        x0,
                        t_eval=ts_refine,
                    )

                    x_learnt = np.transpose(sol_learnt.y).reshape(1, -1, num_indp_var)
                    dic_element["x_learnt"] = x_learnt

                    x_learnt_list.append(dic_element)

        # print(len(x_learnt_list))

        x_learnt_noise_neuron_list.append(x_learnt_list)

    return x_learnt_noise_neuron_list


def post_proscesing_2(coeff_noise_list, param_set, NN_sensitivity, Sample_sensitivity):
    
    
    """
    Discription:
                This function will postprocess the estimated models computed by
                simulation_sensitivity.py module, 
    Args:
        coeff_noise_list: a list containing the estimated coefficients with RK4, iNeuralSINDy, and NeuralSINDy
        param_set: a data class that defines our dynamical system setting
        NN_sensitivity: bool = True/False, if we simulate for different neurons
        Sample_sensitivity: bool = True/False, if we simulate for different samples
        
    Output:
        The outputs are error coeffcients between the reference and different
        algorithms, they are in he form of dictionaries, so one can keep track of
        noise level, sample, algorithm, etc.
        
        er_coef_NNRK_neur_noise_list: ineuralsindy 
        er_coef_RK_neur_noise_list: RK-sindy
        er_coef_NN_neur_noise_list: neuralsindy
        er_curves_NNRK_list: ineuralsindy
        er_curves_RK_list: RK-sindy
        er_curves_NN_list: neuralsindy
    """
    
    
    
    
    num_ind_var = np.shape((coeff_noise_list[0][0]["learned_coeffs"]))[1]
    epsilon = np.ones((1, num_ind_var)) * np.finfo(np.float).eps

    er_coef_NNRK_neur_noise_list = []
    er_coef_RK_neur_noise_list = []
    er_coef_NN_neur_noise_list = []


    er_curves_NNRK_list = []
    er_curves_RK_list = []
    er_curves_NN_list = []
    
    ################################
    ################################
    
    """ Important message: 
        Loading the standard model coefficients: we must do the simulation 
        in advance and save the results somewhere
        
        Here we saved our reference coeffs in param_set.save_model_path directory,
        you can save it anywhere! but make sure to call it correctly because 
        this is the reference model and errors will be wrong otherwise
    """

    with open(
        f"{param_set.save_model_path}" + "coeff_noise_list_specific_setting", "rb"
    ) as fp:
        coeff_ref_list = pickle.load(fp)

    refer_coef = np.round(coeff_ref_list[0]["learned_coeffs"], 2)

    
    
    
    """
    #### Alternativly if your estimated coeffs are good enough to be used in
    #### analysis, you are free to do that, but I do not recommend to do it
    
    if NN_sensitivity is True:
        for k in range(len(coeff_noise_list[0])):    
            if coeff_noise_list[0][k]["neuron"] == 32 and coeff_noise_list[0][k]["useNN"] is True and coeff_noise_list[0][k]["useRK"] is True :
                refer_coef = np.round(coeff_noise_list[0][k]["learned_coeffs"],1)
                print(refer_coef)
            
    if Sample_sensitivity is True:   
        refer_coef = np.round(coeff_noise_list[0][-1]["learned_coeffs"],1)    
    """
    
    #################################
    
    

    for l in range(len(coeff_noise_list)):
        er_coef_NNRK_dict = []
        er_coef_RK_dict = []
        er_coef_NN_dict = []

        er_coef_NNRK_list = []
        er_coef_RK_list = []
        er_coef_NN_list = []

        er_curves_NNRK = np.zeros(
            (len(param_set.num_hidden_neur), param_set.num_indp_var + 1)
        )
        er_curves_RK = np.zeros(
            (len(param_set.num_hidden_neur), param_set.num_indp_var + 1)
        )
        er_curves_NN = np.zeros(
            (len(param_set.num_hidden_neur), param_set.num_indp_var + 1)
        )

        n1 = 0
        n2 = 0
        n3 = 0

        for i in range(len(coeff_noise_list[l])):
            if (
                coeff_noise_list[l][i]["useRK"] is True
                and coeff_noise_list[l][i]["useNN"] is True
            ):
                # if coeff_noise_list[l][i]["noise"] == 0.0:
                # refer_coef = coeff_noise_list[l][i]["learned_coeffs"]

                er_coef_NNRK = {
                    "tech": "NNRK",
                    "noise": coeff_noise_list[l][i]["noise"],
                    "neuron": coeff_noise_list[l][i]["neuron"],
                    "error": epsilon
                    + np.sum(
                        np.abs(
                            np.subtract(
                                refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                            )
                        ),
                        axis=0,
                    ),
                }
                er_coef_NNRK_dict.append(er_coef_NNRK)

                er_coef_NNRK_list.append(
                    np.sum(
                        np.abs(
                            np.subtract(
                                refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                            )
                        ),
                        axis=0,
                    )
                )

                er_curves_NNRK[n1, 0] = int(coeff_noise_list[l][i]["neuron"])
                er_curves_NNRK[n1, 1:] = np.sum(
                    np.abs(
                        np.subtract(
                            refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                        )
                    ),
                    axis=0,
                )
                n1 += 1

            if (
                coeff_noise_list[l][i]["useRK"] is True
                and coeff_noise_list[l][i]["useNN"] is False
            ):
                er_coef_RK = {
                    "tech": "RK",
                    "noise": coeff_noise_list[l][i]["noise"],
                    "neuron": coeff_noise_list[l][i]["neuron"],
                    "error": np.sum(
                        np.abs(
                            np.subtract(
                                refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                            )
                        ),
                        axis=0,
                    ),
                }
                er_coef_RK_dict.append(er_coef_RK)

                er_coef_RK_list.append(
                    np.sum(
                        np.abs(
                            np.subtract(
                                refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                            )
                        ),
                        axis=0,
                    )
                )

                er_curves_RK[n2, 0] = int(coeff_noise_list[l][i]["neuron"])
                er_curves_RK[n2, 1:] = np.sum(
                    np.abs(
                        np.subtract(
                            refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                        )
                    ),
                    axis=0,
                )
                n2 += 1

            if (
                coeff_noise_list[l][i]["useRK"] is False
                and coeff_noise_list[l][i]["useNN"] is True
            ):
                er_coef_NN = {
                    "tech": "NN",
                    "noise": coeff_noise_list[l][i]["noise"],
                    "neuron": coeff_noise_list[l][i]["neuron"],
                    "error": np.sum(
                        np.abs(
                            np.subtract(
                                refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                            )
                        ),
                        axis=0,
                    ),
                }
                er_coef_NN_dict.append(er_coef_NN)

                er_coef_NN_list.append(
                    np.sum(
                        np.abs(
                            np.subtract(
                                refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                            )
                        ),
                        axis=0,
                    )
                )

                er_curves_NN[n3, 0] = int(coeff_noise_list[l][i]["neuron"])
                er_curves_NN[n3, 1:] = np.sum(
                    np.abs(
                        np.subtract(
                            refer_coef, coeff_noise_list[l][i]["learned_coeffs"]
                        )
                    ),
                    axis=0,
                )
                n3 += 1

        er_coef_NNRK_neur_noise_list.append(er_coef_NNRK_dict)
        er_coef_RK_neur_noise_list.append(er_coef_RK_dict)
        er_coef_NN_neur_noise_list.append(er_coef_NN_dict)

        er_curves_NNRK_list.append(er_curves_NNRK)
        er_curves_RK_list.append(er_curves_RK)
        er_curves_NN_list.append(er_curves_NN)

    return (
        er_coef_NNRK_neur_noise_list,
        er_coef_RK_neur_noise_list,
        er_coef_NN_neur_noise_list,
        er_curves_NNRK_list,
        er_curves_RK_list,
        er_curves_NN_list,
    )


##################################
##################################
##################################


def post_proscesing(coeff_noise_list):
    
    """
    This function is similar to post_processing_2 function that are mentioned above,
    but post_processing_2 is more general and I recommend to make use of it,
    This function is made for anlysis of different noise level, 
    the reference coeffs is the one corresponding to noise level=0
    
    args:
        coeff_noise_list: it is a list of dictionaries, element of the list
        stores the values of coeffs corresponding to each algorithm, noise level, etc.
    
    """
    
    
    
    er_coef_NNRK_dict = []
    er_coef_RK_dict = []
    er_coef_NN_dict = []

    er_coef_NNRK_list = []
    er_coef_RK_list = []
    er_coef_NN_list = []

    num_ind_var = np.shape((coeff_noise_list[0]["learned_coeffs"]))[1]
    epsilon = np.ones((1, num_ind_var)) * np.finfo(np.float).eps

    for i in range(len(coeff_noise_list)):
        if (
            coeff_noise_list[i]["useRK"] is True
            and coeff_noise_list[i]["useNN"] is True
        ):
            if coeff_noise_list[i]["noise"] == 0.0:
                refer_coef = coeff_noise_list[i]["learned_coeffs"]

            # np.sum(np.abs(np.subtract(refer_coef, coeff_noise_list[i]["learned_coeffs"])),axis=0)

            er_coef_NNRK = {
                "tech": "NNRK",
                "noise": coeff_noise_list[i]["noise"],
                "error": epsilon
                + np.sum(
                    np.abs(
                        np.subtract(refer_coef, coeff_noise_list[i]["learned_coeffs"])
                    ),
                    axis=0,
                ),
            }

            er_coef_NNRK_dict.append(er_coef_NNRK)

            er_coef_NNRK_list.append(
                np.sum(
                    np.abs(
                        np.subtract(refer_coef, coeff_noise_list[i]["learned_coeffs"])
                    ),
                    axis=0,
                )
            )


        if (
            coeff_noise_list[i]["useRK"] is True
            and coeff_noise_list[i]["useNN"] is False
        ):
            er_coef_RK = {
                "tech": "RK",
                "noise": coeff_noise_list[i]["noise"],
                "error": np.sum(
                    np.abs(
                        np.subtract(refer_coef, coeff_noise_list[i]["learned_coeffs"])
                    ),
                    axis=0,
                ),
            }
            er_coef_RK_dict.append(er_coef_RK)

            er_coef_RK_list.append(
                np.sum(
                    np.abs(
                        np.subtract(refer_coef, coeff_noise_list[i]["learned_coeffs"])
                    ),
                    axis=0,
                )
            )

        if (
            coeff_noise_list[i]["useRK"] is False
            and coeff_noise_list[i]["useNN"] is True
        ):
            er_coef_NN = {
                "tech": "NN",
                "noise": coeff_noise_list[i]["noise"],
                "error": np.sum(
                    np.abs(
                        np.subtract(refer_coef, coeff_noise_list[i]["learned_coeffs"])
                    ),
                    axis=0,
                ),
            }
            er_coef_NN_dict.append(er_coef_NN)

            er_coef_NN_list.append(
                np.sum(
                    np.abs(
                        np.subtract(refer_coef, coeff_noise_list[i]["learned_coeffs"])
                    ),
                    axis=0,
                )
            )

    return (
        np.reshape(er_coef_NNRK_list, (-1, num_ind_var)),
        np.reshape(er_coef_RK_list, (-1, num_ind_var)),
        np.reshape(er_coef_NN_list, (-1, num_ind_var)),
    )
