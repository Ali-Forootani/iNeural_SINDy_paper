import torch
from Functions.root_classes import CompleteNetwork
import torch.nn as nn

import matplotlib.pyplot as plt

from Functions.linear_coeffs_module import (
    rk4th_onestep_SparseId,
    rk4th_onestep_SparseId_non_rational,
)


from Functions.logger_non_rational import Logger


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    estimated_coeffs_K,
    write_iterations=5000,
    max_iterations=50000,
    threshold_iteration=15000,
    threshold_value=0.02,
    RK_timestep=0.05,
    useRK=True,
    useNN=True,
) -> None:
    # for key in func_main_args:
    # if key == "model":
    #    model = func_main_args[key]
    # if key == "train_dataloader":
    #    train_dataloader = func_main_args[key]
    # if key == "test_dataloader":
    #    test_dataloader = func_main_args[key]
    # if key == "optimizer":
    #    optimizer = func_main_args[key]
    # if key == "estimated_coeffs_k":
    #    estimated_coeffs_K = func_main_args[key]
    # if key == "write_iterations":
    #    write_iterations = func_main_args[key]
    # if key == "max_iterations":
    #    max_iterations = func_main_args[key]
    # if key == "threshold_iteration":
    #    threshold_iteration = func_main_args[key]
    # if key == "threshold_value":
    #    threshold_value = func_main_args[key]
    # if key == "RK_timestep":
    #    RK_timestep = func_main_args[key]
    # if key == "useRK":
    #    useRK = func_main_args[key]
    # if key == "useNN":
    #    useNN = func_main_args[key]

    """
    Training loop function:
        args:
            model: DNN module, MLP with SIREN activation function
            train_dataloader: train data set, torch data loader
            test_dataloader: test data set, torch data loader
            optimizer: torch optimizer
            estimated_coeffs_k: a linear NN module that is used as the coeffs
                                this parameter is the main output of the algorithms

            max_iterations: number of epochs
            threshold_iteration: the maximum iteration that we allow our network
                                to be trained without enforcing thresholding

            write_iterations: the iteration that we apply thresholding by this
                                condition: iteration % write_iterations == 0 & iteration > threshold_iteration


        outputs:
            loss_values: total loss in one epoch which is the summation of three terms, e.g.
                        NN_loss +  loss_time_coeff + RK4_loss
            loss_values_NN: DNN loss through the iteration
            loss_values_Coeff: subtraction of time derivative and (library terms * coeffs) through the training
            loss_values_RK4:  loss values corresponding to RK4 integration scheme
            coeff_track_list: tracking the coeffs through the training loop


    """

    model.train()
    print(f"Model set-up useRK: {useRK}, useNN: {useNN}!")

    loss_fn = nn.MSELoss()
    loss_values = []
    loss_values_NN = []
    loss_values_Coeff = []
    loss_values_RK4 = []
    loss_mse_test = []
    coeff_NN_list = []
    coeff_RK_list = []
    coeff_NNRK_list = []
    coeff_track_list = []

    lr_red = 2

    exp_ID: str = None
    log_dir: str = "log_files"

    logger = Logger(exp_ID, log_dir)

    for iteration in range(max_iterations):
        running_loss = 0.0
        running_loss_NN = 0.0
        running_loss_Coeff = 0.0
        running_loss_RK4 = 0.0

        loss_prediction_target = torch.autograd.Variable(
            torch.tensor([0.0], requires_grad=True)
        ).to(device)
        loss_RK4 = torch.autograd.Variable(torch.tensor([0.0], requires_grad=True)).to(
            device
        )
        loss_time_coeff = torch.autograd.Variable(
            torch.tensor([0.0], requires_grad=True)
        ).to(device)

        optimizer.zero_grad()

        for i in range(len(train_dataloader)):
            for batch_idx, train_sample in enumerate(train_dataloader[i]):
                "unpacking the train data"
                (data_train, target_train) = train_sample

                #######################################

                """loading the data to GPU """

                data_train = data_train.to(device)
                target_train = target_train.to(device)

                ########################################

                """ feeding the data to DNN model"""
                (
                    prediction,
                    time_derivs_k,
                    thetas_k,
                    output_nn_k,
                ) = model(data_train)

                ##########################################
                """multiplying the library by coeffs, right side of diff equation. i.e.:
                    dx/dt = \theta * coeffs
                """

                K_X = thetas_k @ torch.t(model.estimated_coeffs_k.linear.weight)

                total_coeffs = K_X

                ##########################################
                # =================== Loss functions ===============
                """constructing loss functions, rk4, NN, etc."""

                if useRK:
                    RK4_pred = rk4th_onestep_SparseId_non_rational(
                        target_train[:-1],
                        model.library_k,
                        model,
                        timestep=RK_timestep,
                        t=-1,
                    )

                    loss_RK4 += (1 / RK_timestep) * torch.mean(
                        (target_train[1:] - RK4_pred) ** 2
                    )

                else:
                    RK4_pred = rk4th_onestep_SparseId_non_rational(
                        prediction[:-1],
                        model.library_k,
                        model,
                        timestep=RK_timestep,
                        t=-1,
                    )
                    loss_RK4 += (1 / RK_timestep) * torch.mean(
                        (prediction[1:] - RK4_pred) ** 2
                    )

                loss_time_coeff += loss_fn(
                    time_derivs_k.reshape(-1, total_coeffs.shape[-1]), total_coeffs
                )

                loss_prediction_target += torch.mean((prediction - target_train) ** 2)
        
        
        
        """defining the total loss in the training loop, \mu define our training algorithm  
            \mu_1 * loss_prediction_target + \mu_2 * loss_time_coeff + \mu_3 * loss_RK4
        """

        if useRK and not useNN:
            Total_loss = 1 * loss_RK4

        elif useNN and not useRK:
            Total_loss = 1 * loss_prediction_target + 1e-1 * loss_time_coeff

        elif useRK and useNN:
            Total_loss = (
                1 * loss_prediction_target + 1e-1 * loss_time_coeff + 1e-1 * loss_RK4
            )

        coeff_track_list.append(model.estimated_coeffs_k.linear.weight.detach().clone())

        Total_loss.backward()

        optimizer.step()

        running_loss += Total_loss.item()
        running_loss_Coeff += loss_time_coeff.item()
        running_loss_NN += loss_prediction_target.item()
        running_loss_RK4 += loss_RK4.item()

        "tracking the loss function within the training loop"
        loss_values.append(running_loss)
        loss_values_NN.append(running_loss_NN)
        loss_values_Coeff.append(running_loss_Coeff)
        loss_values_RK4.append(running_loss_RK4)
        
        # ================ Logging ===========================
        
        """looging the results of training and printing on the screen """

        logger(
            iteration,
            loss_values,
            loss_values_Coeff,
            loss_values_NN,
            loss_values_RK4,
            model.estimated_coeffs_k.linear.weight,
            write_iterations,
            threshold_iteration,
        )

        """enforcing sparsity by applying threshold after some iteration """

        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            """you can include vaidation here if you would like:"""
           
            
               
            # ================== Sparsity update =============
            

            """masking the coefficients and re-setting the optimizers """

            if iteration > threshold_iteration and iteration % write_iterations == 0:
                Wk = model.estimated_coeffs_k.linear.weight.detach().clone()
                tl_value = threshold_value
                Mask_Wk = (Wk.abs() > tl_value).type(torch.float)
                model.estimated_coeffs_k.linear.weight = torch.nn.Parameter(
                    Wk * Mask_Wk
                )

                """setting the hook with new weights """

                model.estimated_coeffs_k.linear.weight.register_hook(
                    lambda grad: grad.mul_(Mask_Wk)
                )

                optimizer = torch.optim.Adam(
                    [
                        {
                            "params": model.func_approx.parameters(),
                            "lr": 5e-6 / lr_red,
                            "weight_decay": 0,
                        },
                        # {
                        #    "params": model.estimated_coeffs.parameters(),
                        #    "lr": 5e-6 / lr_red,
                        #    "weight_decay": 0,
                        #    },
                        {
                            "params": model.estimated_coeffs_k.parameters(),
                            "lr": 1e-2 / lr_red,
                            "weight_decay": 0,
                        },
                        #                      {'params': model.library.parameters(), 'lr': 5e-2, 'weight_decay': 1e-4},
                    ]
                )

                lr_red = 1 * lr_red

                ###########################################
                ###########################################
                ###########################################
                ###########################################
                ###########################################
                ###########################################

    return (
        loss_values,
        loss_values_NN,
        loss_values_Coeff,
        loss_values_RK4,
        coeff_track_list,
    )
