#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:48:18 2022

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
from Functions.utiles import initial_cond_generation
from Functions.linear_coeffs_module import CoeffsNetwork, CoeffsDictionaryRational

######################
from Functions.root_classes_non_rational import CompleteNetwork
from Functions.training_non_rational import train
from Functions.model_recovery_func import (
    model_recovery,
    model_recovery_single_noise_level,
)
import Functions.plot_config_file

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(" We are using a " + device + " for the simulation")
#####################
#####################
### Making a list of initial conditions

# @dataclass
# class Parameters_setting:


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

    t_max = 10
    t_min = 0
    num_samples = 801
    num_init_cond = 2
    num_indp_var = 2
    min_init_cond = 1
    max_init_cond = 4
    poly_order = 3
    fun_scaling_factor = 1
    max_iterations = 30000
    write_iterations = 5000
    threshold_iteration = 15000
    threshold_value = 0.05
    shuffle = False
    save_model_path: str = "./Results/Cubic_Oscilator/"
    add_noise: bool = True
    noise_level = [0.00]
    useRK = [True, False]
    useNN = [True, False]
    tikz_save: bool = False
    name_pic_loss: str = "Cubic_Oscilator_loss_noise_"
    name_pic_data: str = "Cubic_Oscilator_training_data_noise_"
    name_pic_comparison: str = "Cubic_Oscilator_noise_"


param_set = ParametersSetting()
ts = torch.linspace(param_set.t_min, param_set.t_max, param_set.num_samples)
os.makedirs(os.path.dirname(param_set.save_model_path), exist_ok=True)
RK_timestep = ts[1] - ts[0]


###################
###################
###################

"""
Defining the DNN network with customized activation function
"""

network = Siren(
    int(param_set.num_indp_var + 1), [32, 32, 32], int(param_set.num_indp_var)
).to(device)
loss_fn = nn.MSELoss()

####################
####################
####################

"""
time_deriv_coef: here it is the concept, dx/dt = time_deriv_coef * \theta * coeffs 
"""

time_deriv_coef = 2 / (param_set.t_max - param_set.t_min)



##############################


"""
To compute the number of terms in the customized library
"""


combinations_main = _combinations(
    param_set.num_indp_var,
    param_set.poly_order,
    include_interaction=True,
    include_bias=True,
    interaction_only=False,
)

coef_dim_row = sum(1 for _ in combinations_main)



start = time.time()

#######################
#######################


""" creating list of initial conditions or inserting your own choices """

list_initial_conditions = initial_cond_generation(
    param_set.num_init_cond,
    param_set.num_indp_var,
    param_set.min_init_cond,
    param_set.max_init_cond,
)


list_initial_conditions = [[2, 2], [-2, -2]]


#######################
#######################

""" Some useful syntaxes in torch library
To go from np.array to cpu Tensor, use torch.from_numpy().
To go from cpu Tensor to gpu Tensor, use .cuda().
To go from a Tensor that requires_grad to one that does not, use .detach() (in your case, your net output will most likely requires gradients and so it’s output will need to be detached).
To go from a gpu Tensor to cpu Tensor, use .cpu().
To go from a cpu Tensor to np.array, use .numpy().
"""



"""The following list store the estimed coeffs, so you can print them to see
    the simulation results
"""

coeff_noise_list = []
dic_coef_track_list = []
use_RK = [True, False]
use_NN = [True, False]


####################
####################


for i in range(len(param_set.noise_level)):
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

                model = (
                    CompleteNetwork(
                        network,
                        library_k,
                        estimated_coeffs_k,
                    )
                    .float()
                    .to(device)
                )

                optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model.func_approx.parameters(),
                            "lr": 1e-4,
                            "weight_decay": 0,
                        },
                        {
                            "params": model.estimated_coeffs_k.parameters(),
                            "lr": 1e-3,
                            "weight_decay": 0,
                        },
                    ]
                )
                
                """Creating an instance as a dynamical system, passing 
                    a name as the dynamical system, e.g. 2_D_Oscilator
                """

                print("\n")
                y = Factory_Dyn_Sys(
                    ts,
                    list_initial_conditions,
                    param_set.fun_scaling_factor,
                    "Cubic Damped Oscillator",
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
                
                # ============ Creating data loader for torch
                
                
                train_dataloader, test_dataloader = train_test_spliting_dataset(
                    list_t_scaled_main,
                    list_initial_cond_main,
                    list_u_original,
                    device,
                    batch_size=2500,
                    split_value=0.9,
                    shuffle=False,
                )
                
                """ Calling training function """
                
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
                    useRK=param_set.useRK[k],
                    useNN=param_set.useNN[j],
                )
               
                """ putting the results of the simulation in a dictionary """

                dic_coef_track = {
                    "noise": param_set.noise_level[i],
                    "useRK": param_set.useRK[k],
                    "useNN": param_set.useNN[j],
                    "coeff_list_track": coeff_track_list,
                }

                dic_coeff_noise = {
                    "noise": param_set.noise_level[i],
                    "useRK": param_set.useRK[k],
                    "useNN": param_set.useNN[j],
                    "learned_coeffs": model.estimated_coeffs_k.linear.weight.detach()
                    .clone()
                    .t()
                    .cpu()
                    .numpy(),
                }
                coeff_noise_list.append(dic_coeff_noise)

                dic_coef_track_list.append(dic_coef_track)

print(len(dic_coef_track_list))

###########################
###########################

""" To recover the model from estimated coeffs: Integral (library * coeffs) = x(t)"""

dict_solution = model_recovery_single_noise_level(
    coeff_noise_list,
    list_initial_conditions,
    param_set.t_max,
    param_set.poly_order,
    param_set.num_indp_var,
    t_step=0.025,
)


####################
#################### Plotting the curves
plt.figure(6, figsize=(5, 3))
for i in range(int(param_set.num_init_cond)):
    for j in range(int(param_set.num_indp_var)):
        plt.plot(ts, list_u_original[i][:, j])
    # im = ax.scatter(list_t_scaled_main[0], list_u_original[i][:,1], marker="+", s=10)
plt.xlabel("t", fontweight="bold", fontsize=20)
plt.ylabel("$x(t)$", fontweight="bold", fontsize=20)
# plt.title("Cubic Oscillator")
# plt.savefig(param_set.save_model_path + '2_D_Oscillatory_training_data_noise_'+str(param_set.noise_level[0])+'.png')

plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_data
    + str(param_set.noise_level[0])
    + ".png",
    transparent=True,
)


plt.show()
####################
end = time.time()


######################################


""" Saving the result in the same directory """
with open(
    f"{param_set.save_model_path}" + "coeff_noise_list" + str(param_set.noise_level[0]),
    "wb",
) as fp:
    pickle.dump(coeff_noise_list, fp)

with open(
    f"{param_set.save_model_path}" + "list_u_original" + str(param_set.noise_level[0]),
    "wb",
) as fp:
    pickle.dump(list_u_original, fp)

with open(
    f"{param_set.save_model_path}" + "dict_solution" + str(param_set.noise_level[0]),
    "wb",
) as fp:
    pickle.dump(dict_solution, fp)

with open(
    f"{param_set.save_model_path}"
    + "true_data_noise_free"
    + str(param_set.noise_level[0]),
    "wb",
) as fp:
    pickle.dump(true_data_noise_free, fp)

with open(f"{param_set.save_model_path}" + "ts", "wb") as fp:
    pickle.dump(ts, fp)

with open(
    f"{param_set.save_model_path}"
    + "dic_coef_track_list"
    + str(param_set.noise_level[0]),
    "wb",
) as fp:
    pickle.dump(true_data_noise_free, fp)


#####################################
#####################################

"""
To go from np.array to cpu Tensor, use torch.from_numpy().
To go from cpu Tensor to gpu Tensor, use .cuda().
To go from a Tensor that requires_grad to one that does not, use .detach() (in your case, your net output will most likely requires gradients and so it’s output will need to be detached).
To go from a gpu Tensor to cpu Tensor, use .cpu().
To go from a cpu Tensor to np.array, use .numpy().
"""

print("\n")
print(f"The elapsed time is: {end-start} " + "seconds \n")
print(
    f"The ODE coefficients by our approach are:\n {model.estimated_coeffs_k.linear.weight.detach().clone().t().cpu().numpy()}"
)

print(f"The ODE coefficients by different approaches are:\n {coeff_noise_list}")


"""Saving the coefficients in the directory"""
with open(
    f"{param_set.save_model_path}" + "coeff_noise_list_specific_setting", "wb"
) as fp:
    pickle.dump(coeff_noise_list, fp)


########################################
########################################

""" Plotting the results, loss functions"""

plt.figure(1, figsize=(20, 3))
plt.subplot(1, 4, 1)
plt.semilogy(loss_values)
plt.grid(True, which="both")
plt.xlabel("Total loss", fontsize=30)


plt.subplots_adjust(left=0.15)
plt.subplot(1, 4, 2)
plt.semilogy(loss_values_NN)
plt.grid(True, which="both")
plt.xlabel("NN loss", fontsize=30)


plt.subplots_adjust(left=0.15)
plt.subplot(1, 4, 3)
plt.semilogy(loss_values_Coeff)
plt.grid(True, which="both")
plt.xlabel("Derivative loss", fontsize=30)

plt.subplots_adjust(left=0.15)
plt.subplot(1, 4, 4)
plt.semilogy(loss_values_RK4)
plt.grid(True, which="both")
# plt.xticks(fontname="serif" ,fontsize=8)
plt.xlabel("Rk4 loss", fontsize=30)
plt.subplots_adjust(left=0.15)

if param_set.tikz_save:
    tikzplotlib.save(param_set.save_model_path + "Cubic_Oscilator_loss.tex")
# plt.savefig(param_set.save_model_path + '2_D_Oscillatory_loss_noise_'+str(param_set.noise_level[0])+'.png')

plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_loss
    + str(param_set.noise_level[0])
    + str(param_set.num_samples)
    + ".png"
)


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


#########################################
#########################################


with open(f"{param_set.save_model_path}" + "dict_solution0.0", "rb") as fp:
    dict_solution = pickle.load(fp)
    # x_ineural = dict_solution[0]['x_learnt']

    # x_truth = x_ineural

with open(f"{param_set.save_model_path}" + "list_u_original0.0", "rb") as fp:
    list_u_original = pickle.load(fp)

with open(f"{param_set.save_model_path}" + "true_data_noise_free0.0", "rb") as fp:
    true_data_noise_free = pickle.load(fp)


fig, axs = plt.subplots(1, 3, sharey=False, figsize=(10, 3.5))

fig_parameters = {"linewidth": 3}
line_style = {"linestyle": "--"}


plt.rcParams.update(
    {
        "font.size": 20,
    }
)


for i in range(len(dict_solution)):
    if dict_solution[i]["useRK"] is True and dict_solution[i]["useNN"] is True:
        axs[0].plot(
            list_u_original[0][:, 0],
            list_u_original[0][:, 1],
            "ko",
            **fig_parameters,
            label="noisy measurement",
        )
        axs[0].plot(
            true_data_noise_free[0][:, 0],
            true_data_noise_free[0][:, 1],
            color=colors[0],
            **fig_parameters,
            label=r"truth model",
        )
        axs[0].plot(
            dict_solution[i]["x_learnt"][0, :, 0],
            dict_solution[i]["x_learnt"][0, :, 1],
            color=colors[1],
            linestyle="--",
            **fig_parameters,
            label=r"\texttt{iNeural-SINDy}",
        )
        axs[0].set(ylabel="$x_2$")

    if dict_solution[i]["useRK"] is True and dict_solution[i]["useNN"] is False:
        axs[1].plot(
            list_u_original[0][:, 0],
            list_u_original[0][:, 1],
            "ko",
            **fig_parameters,
        )
        axs[1].plot(
            true_data_noise_free[0][:, 0],
            true_data_noise_free[0][:, 1],
            color=colors[0],
            **fig_parameters,
        )
        axs[1].plot(
            dict_solution[i]["x_learnt"][0, :, 0],
            dict_solution[i]["x_learnt"][0, :, 1],
            color=colors[2],
            linestyle="--",
            **fig_parameters,
            label=r"\texttt{RK4-SINDy}",
        )

    if dict_solution[i]["useRK"] is False and dict_solution[i]["useNN"] is True:
        axs[2].plot(
            list_u_original[0][:, 0],
            list_u_original[0][:, 1],
            "ko",
            **fig_parameters,
        )
        axs[2].plot(
            true_data_noise_free[0][:, 0],
            true_data_noise_free[0][:, 1],
            color=colors[0],
            **fig_parameters,
        )
        axs[2].plot(
            dict_solution[i]["x_learnt"][0, :, 0],
            dict_solution[i]["x_learnt"][0, :, 1],
            color=colors[3],
            linestyle="--",
            **fig_parameters,
            label=r"\texttt{DeePyMoD}",
        )


for ax in axs.flat:
    ax.set(xlabel=r"$x_1$")
    # ax.set(ylabel=r'$x_2$')

lines, labels = [], []
for ax in fig.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)


axs[0].set_xlim((-2.5, 2.5))
axs[0].set_ylim((-2.5, 2.5))

axs[1].set_xlim((-2.5, 2.5))
axs[1].set_ylim((-2.5, 2.5))

axs[2].set_xlim((-2.5, 2.5))
axs[2].set_ylim((-2.5, 2.5))


fig.legend(
    lines,
    labels,
    loc="center",
    ncol=3,
    bbox_to_anchor=(0.53, 1.07),
    bbox_transform=fig.transFigure,
    fontsize=20,
    frameon=True,
)

plt.tight_layout()


plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_comparison
    + str(param_set.noise_level[0])
    + str(param_set.num_samples)
    + "_threshold_"
    + str(param_set.threshold_value)
    + ".pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_comparison
    + str(param_set.noise_level[0])
    + str(param_set.num_samples)
    + "_threshold_"
    + str(param_set.threshold_value)
    + ".png",
    bbox_inches="tight",
    dpi=300,
)

plt.show()


###################################
###################################
###################################


plt.rcParams.update(
    {
        "font.size": 20,
    }
)


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

coef_size = np.size(
    model.estimated_coeffs_k.linear.weight.detach().clone().t().cpu().numpy()
)


track_coef_matrix_1_NNRK = torch.zeros((param_set.max_iterations, int(coef_size / 2)))
track_coef_matrix_2_NNRK = torch.zeros((param_set.max_iterations, int(coef_size / 2)))

track_coef_matrix_1_RK = torch.zeros((param_set.max_iterations, int(coef_size / 2)))
track_coef_matrix_2_RK = torch.zeros((param_set.max_iterations, int(coef_size / 2)))

track_coef_matrix_1_NN = torch.zeros((param_set.max_iterations, int(coef_size / 2)))
track_coef_matrix_2_NN = torch.zeros((param_set.max_iterations, int(coef_size / 2)))

for i in range(param_set.max_iterations):
    for j in range(2):
        track_coef_matrix_1_NNRK[i, :] = dic_coef_track_list[0]["coeff_list_track"][i][
            0
        ]
        track_coef_matrix_2_NNRK[i, :] = dic_coef_track_list[0]["coeff_list_track"][i][
            1
        ]

        track_coef_matrix_1_RK[i, :] = dic_coef_track_list[1]["coeff_list_track"][i][0]
        track_coef_matrix_2_RK[i, :] = dic_coef_track_list[1]["coeff_list_track"][i][1]

        track_coef_matrix_1_NN[i, :] = dic_coef_track_list[2]["coeff_list_track"][i][0]
        track_coef_matrix_2_NN[i, :] = dic_coef_track_list[2]["coeff_list_track"][i][1]

fig, axs = plt.subplots(2, 3, figsize=(10, 5))



line_style = {"linestyle": "--"}

for l in range(1):
    axs[0, 0].plot(
        track_coef_matrix_1_NNRK[:15000, l],
        color=colors[1],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
        label=r"\texttt{iNeural-SINDy}",
    )
    axs[0, 0].set(ylabel="Coeffs for $x_1$")

    axs[1, 0].plot(
        track_coef_matrix_2_NNRK[:15000, l],
        color=colors[1],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
    axs[1, 0].set(ylabel="Coeffs for $x_2$")
    axs[1, 0].set(xlabel="Iteration")

    axs[0, 1].plot(
        track_coef_matrix_1_RK[:15000, l],
        color=colors[2],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
        label=r"\texttt{RK4-SINDy}",
    )
    
    
    axs[1, 1].plot(
        track_coef_matrix_2_RK[:15000, l],
        color=colors[2],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
    
    
    axs[1, 1].set(xlabel="Iteration")

    axs[0, 2].plot(
        track_coef_matrix_1_NN[:15000, l],
        color=colors[3],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
        label=r"\texttt{DeePyMoD}",
    )
    
    
    axs[1, 2].plot(
        track_coef_matrix_2_NN[:15000, l],
        color=colors[3],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
    
    
    axs[1, 2].set(xlabel="Iteration")


for l in range(1, int(coef_size / 2)):
    axs[0, 0].plot(
        track_coef_matrix_1_NNRK[:15000, l],
        color=colors[1],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
    
    
    axs[1, 0].plot(
        track_coef_matrix_2_NNRK[:15000, l],
        color=colors[1],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )

    axs[0, 1].plot(
        track_coef_matrix_1_RK[:15000, l],
        color=colors[2],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
   
    
    axs[1, 1].plot(
        track_coef_matrix_2_RK[:15000, l],
        color=colors[2],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
    
    
    axs[0, 2].plot(
        track_coef_matrix_1_NN[:15000, l],
        color=colors[3],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
    
    
    axs[1, 2].plot(
        track_coef_matrix_2_NN[:15000, l],
        color=colors[3],
        **line_style,
        markersize=5,
        markevery=1000,
        **fig_parameters,
    )
   


lines, labels = [], []
for ax in fig.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

fig.legend(
    lines,
    labels,
    loc="center",
    bbox_to_anchor=(0.53, 1.02),
    ncol=len(labels),
    bbox_transform=fig.transFigure,
)
plt.tight_layout()


plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_comparison
    + str(param_set.noise_level[0])
    + "_Coefficients_"
    + "_threshold_"
    + str(param_set.threshold_value)
    + ".pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_comparison
    + str(param_set.noise_level[0])
    + "_Coefficients_"
    + "_threshold_"
    + str(param_set.threshold_value)
    + ".png",
    bbox_inches="tight",
    dpi=300,
)



###################################
###################################
###################################
"""Plotting non-zero coefficients"""

fig, axs = plt.subplots(1, 4, figsize=(30, 8))
# fig.suptitle("Cubic Oscillator", fontsize=30, y=0.92)


axs[0].plot(
    track_coef_matrix_1_NNRK[:7500, 6],
    color=colors[1],
    marker="^",
    linestyle="-",
    markersize=10,
    markevery=500,
    linewidth=5,
    label="iNeuralSINDy",
)
axs[0].plot(
    track_coef_matrix_1_RK[:7500, 6],
    color=colors[2],
    marker="8",
    linestyle="--",
    markersize=10,
    markevery=500,
    linewidth=5,
    label="RK4SINDy",
)
axs[0].plot(
    track_coef_matrix_1_NN[:7500, 6],
    color=colors[3],
    marker="s",
    linestyle="-.",
    markersize=10,
    markevery=500,
    linewidth=5,
    label="NeuralSINDy",
)
axs[0].set(ylabel="Coefficients")
axs[0].set(xlabel="Iteration")


axs[1].plot(
    track_coef_matrix_1_NNRK[:7500, 9],
    color=colors[1],
    marker="^",
    linestyle="-",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[1].plot(
    track_coef_matrix_1_RK[:7500, 9],
    color=colors[2],
    marker="8",
    linestyle="--",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[1].plot(
    track_coef_matrix_1_NN[:7500, 9],
    color=colors[3],
    marker="s",
    linestyle="-.",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[1].set(xlabel="Iteration")


axs[2].plot(
    track_coef_matrix_2_NNRK[:7500, 6],
    color=colors[1],
    marker="^",
    linestyle="-",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[2].plot(
    track_coef_matrix_2_RK[:7500, 6],
    color=colors[2],
    marker="8",
    linestyle="--",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[2].plot(
    track_coef_matrix_2_NN[:7500, 6],
    color=colors[3],
    marker="s",
    linestyle="-.",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[2].set(xlabel="Iteration")


axs[3].plot(
    track_coef_matrix_2_NNRK[:7500, 9],
    color=colors[1],
    marker="^",
    linestyle="-",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[3].plot(
    track_coef_matrix_2_RK[:7500, 9],
    color=colors[2],
    marker="8",
    linestyle="--",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[3].plot(
    track_coef_matrix_2_NN[:7500, 9],
    color=colors[3],
    marker="s",
    linestyle="-.",
    markersize=10,
    markevery=500,
    linewidth=5,
)
axs[3].set(xlabel="Iteration")


lines, labels = [], []
for ax in fig.axes:
    line, label = ax.get_legend_handles_labels()
    lines.extend(line)
    labels.extend(label)

fig.legend(
    lines,
    labels,
    loc="center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=len(labels),
    bbox_transform=fig.transFigure,
    fontsize=25,
)


plt.savefig(
    param_set.save_model_path
    + param_set.name_pic_comparison
    + str(param_set.noise_level[0])
    + "_Slected_Coefficients_"
    + "_threshold_"
    + str(param_set.threshold_value)
    + ".png"
)


##################################
##################################
##################################
