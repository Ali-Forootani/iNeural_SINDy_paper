import torch
from root_classes import complete_network
import torch.nn as nn

import matplotlib.pyplot as plt

from linear_coeffs_module import (rk4th_onestep_SparseId,
                                  rk4th_onestep_SparseId_non_rational)


from logger import Logger

# A syntax to reformat the text body of the code-- we should execute it on 
# the terminal 

# black training_rational.py 






def train_2(
    func_arguments
) -> None:
    
    for key in func_arguments:
        if key == "model":
            model = func_arguments[key]
        if key == "train_dataloader":
            train_dataloader = func_arguments[key]
        if key == "test_dataloader":
            test_dataloader = func_arguments[key]
        if key == "optimizer":
            optimizer = func_arguments[key]
        if key == "estimated_coeffs":
            estimated_coeffs = func_arguments[key]
        if key == "estimated_coeffs_k":
            estimated_coeffs_K = func_arguments[key]
        if key == "write_iterations":
            write_iterations = func_arguments[key]
        if key == "max_iterations":
            max_iterations = func_arguments[key]
        if key == "threshold_iteration":
            threshold_iteration = func_arguments[key]
        if key == "threshold_value":
            threshold_value = func_arguments[key]
        if key == "RK_timestep":
            RK_timestep = func_arguments[key]
        if key == "useOnlyRK":
            useOnlyRK = func_arguments[key]
        if key == "useOnlyNN":
            useOnlyNN = func_arguments[key]
        
    model.train()

    loss_fn = nn.MSELoss()
    loss_values = []
    loss_values_NN = []
    loss_values_Coeff = []
    loss_values_RK4 = []
    loss_mse_test = []
    lr_red = 2
    
    exp_ID: str = None
    log_dir: str = "log_files"
    
    
    logger = Logger(exp_ID, log_dir)

    for iteration in range(max_iterations):

        running_loss = 0.0
        running_loss_NN = 0.0
        running_loss_Coeff = 0.0
        running_loss_RK4 = 0.0
        
        
        
        loss_prediction_target = torch.autograd.Variable(torch.tensor([0.],requires_grad=True))
        loss_RK4 = torch.autograd.Variable(torch.tensor([0.],requires_grad=True))
        loss_time_coeff = torch.autograd.Variable(torch.tensor([0.],requires_grad=True))  
        
        optimizer.zero_grad()
        
        for i in range(len(train_dataloader)):
            
            
            
            for batch_idx, train_sample in enumerate(train_dataloader[i]):
                
                data_train, target_train = train_sample
                
                
                
                
                (
                    prediction,
                    time_derivs,
                    thetas,
                    output_nn,
                    time_derivs_k,
                    thetas_k,
                    output_nn_k,
                ) = model(data_train)
                
                
                Num = thetas @ torch.t(model.estimated_coeffs.numerator.weight)

                
                Den = thetas[:, 1:] @ torch.t(model.estimated_coeffs.denominator.weight)
                
                
                K_X = thetas_k @ torch.t(model.estimated_coeffs_k.linear.weight)
                
                
                
                
                output_coeff_rational = Num / (1 + Den)


                #total_coeffs = output_coeff_rational + K_X
                
                total_coeffs = K_X
                
                if useOnlyRK:
                    RK4_pred = rk4th_onestep_SparseId_non_rational(target_train[:-1],
                                                               model.library_k,
                                                               model,timestep = RK_timestep,t= -1)
                    
                    loss_RK4 += torch.mean((target_train[1:] - RK4_pred) ** 2)
                    
                    
                    
                else:
                    RK4_pred = rk4th_onestep_SparseId_non_rational(prediction[:-1],
                                                               model.library_k,
                                                               model,timestep = RK_timestep,t= -1)
                    loss_RK4 += torch.mean((prediction[1:] - RK4_pred) ** 2)
                    
                
                loss_time_coeff += loss_fn(time_derivs, total_coeffs)
                #loss_RK4 += torch.mean((prediction[1:] - RK4_pred) ** 2)
                loss_prediction_target += torch.mean((prediction - target_train) ** 2)
                
                
        if useOnlyRK:
            Total_loss =  1 * loss_RK4
            
        elif useOnlyNN:
            
            Total_loss = 1 * loss_prediction_target + 0.1 * loss_time_coeff
        else:
            
            
            Total_loss = 1 * loss_prediction_target + 0.1 * loss_time_coeff + 1 * loss_RK4
        
       
        
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
        
        
        logger(
            iteration,
            loss_values,
            loss_values_Coeff,
            loss_values_NN,
            loss_values_RK4,
            model.estimated_coeffs_k.linear.weight,
            model.estimated_coeffs.numerator.weight,
            model.estimated_coeffs.denominator.weight,
            write_iterations,
            threshold_iteration,
            #MSE_test=mse_test,
        )
        
    
        if iteration % write_iterations == 0:
        
            # ================== Validation costs ================
            # with torch.no_grad():
                #    batch_mse_test = torch.zeros(
                #        (n_features, len(test_dataloader)), device = device
                #    )
                #    for batch_idx, test_sample in enumerate(test_dataloader):
                    #        data_test, target_test = test_sample
                    
                    #        prediction_test = model.func_approx(data_test)[0]
                    
                    #        mse_test = torch.mean((prediction_test - target_test)**2)
                    
                    # batch_mse_test[:, batch_idx] = torch.mean(
                    #    (prediction_test - target_test) ** 2, dim=-2
                    # )  # loss per output
                    # mse_test = batch_mse_test.cpu().detach().mean(dim=-1)
                    
                    # loss_mse_test.append(mse_test)
                    # print(estimated_coeffs.linear.weight)
                    
                    # ================== Sparsity update =============
                    # Updating sparsity
                    # update_sparsity = sparsity_scheduler(
                    #    iteration, torch.sum(mse_test), model, optimizer
                    # )
                    
                    if iteration > threshold_iteration and iteration % write_iterations == 0:
                        
                        #Ws = model.estimated_coeffs.numerator.weight.detach().clone()
                        #print('='*20)
                        #print(Ws)
                        #print('='*20)
                        
                        # Ws = model.estimated_coeffs.linear.weight
                        
                        Wn = model.estimated_coeffs.numerator.weight.detach().clone()
                        Wd = model.estimated_coeffs.denominator.weight.detach().clone()
                        
                        Wk = model.estimated_coeffs_k.linear.weight.detach().clone()
                        
                        tl_value = threshold_value
                        
                        Mask_Wn = (Wn.abs() > tl_value).type(torch.float)
                        Mask_Wd = (Wd.abs() > tl_value).type(torch.float)
                        Mask_Wk = (Wk.abs() > tl_value).type(torch.float)
                        
                        model.estimated_coeffs.numerator.weight = torch.nn.Parameter(
                            Wn * Mask_Wn
                            )
                        model.estimated_coeffs.denominator.weight = torch.nn.Parameter(
                            Wd * Mask_Wd
                            )
                        model.estimated_coeffs_k.linear.weight = torch.nn.Parameter(
                            Wk * Mask_Wk
                            )
                        
                        model.estimated_coeffs.numerator.weight.register_hook(
                            lambda grad: grad.mul_(Mask_Wn)
                            )
                        model.estimated_coeffs.denominator.weight.register_hook(
                            lambda grad: grad.mul_(Mask_Wd)
                            )
                        model.estimated_coeffs_k.linear.weight.register_hook(
                            lambda grad: grad.mul_(Mask_Wk)
                            )
                        
                        #### 'lr': 5e-5/lr_red, 'lr': 5e-5/lr_red
                        
                        optimizer = torch.optim.Adam(
                            [
                                {
                                    "params": model.func_approx.parameters(),
                                    "lr": 5e-6 / lr_red,
                                    "weight_decay": 0,
                                    },
                                {
                                    "params": model.estimated_coeffs.parameters(),
                                    "lr": 5e-6 / lr_red,
                                    "weight_decay": 0,
                                    },
                                {
                                    "params": model.estimated_coeffs_k.parameters(),
                                    "lr": 1e-2 / lr_red,
                                    "weight_decay": 0,
                                    },
                                #                      {'params': model.library.parameters(), 'lr': 5e-2, 'weight_decay': 1e-4},
                                ]
                            )
                        
                        lr_red = 1 * lr_red
                        
                        #print("The weights during the training are:")
                        
                        # print(model.estimated_coeffs.linear.weight.grad)
                        
                        #print("Numerator============================")
                        #print(model.estimated_coeffs.numerator.weight)
                        
                        #print("Denominator============================")
                        #print(model.estimated_coeffs.denominator.weight)
                        
                        #print("KKKKKKKKKKKK============================")
                        #print(model.estimated_coeffs_k.linear.weight)
                        
                        #print(loss_values)
                        
                        #print("**********************************")
                        
                        #bvg=torch.sum(torch.abs(model.estimated_coeffs_k.linear.weight), dim=0)
                        
                        #print("**********************************")
                        #print("**********************************")
                        #print("**********************************")
                        #print(bvg)
                        
                        #logger(
                        #    iteration,
                        #    loss_values,
                        #    loss_values_Coeff,
                        #    loss_values_NN,
                        #    model.estimated_coeffs_k.linear.weight,
                        #    model.estimated_coeffs.numerator.weight,
                        #    model.estimated_coeffs.denominator.weight,
                            #MSE_test=mse_test,
                        #)
                        
                        
                        
                        ###########################################
                        ###########################################
                        ###########################################
                        ###########################################
                        ###########################################
                        ###########################################
                        
                        
                                        


    return loss_values, loss_values_NN, loss_values_Coeff, loss_values_RK4



