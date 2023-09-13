



import torch
from root_classes import complete_network
import torch.nn as nn






def train(
    model: complete_network,
    n_features,
    train_dataloader,
    test_dataloader,
    optimizer,
    estimated_coeffs,
    device,
    write_iterations,
    max_iterations
) -> None:
    
    
    model.train()
    
    #n_features = dataset[:][0].shape[1]
    
    loss_fn = nn.MSELoss()

    
    loss_values = []
    loss_values_NN = []
    loss_values_Coeff = []
    
    lr_red = 2


    for iteration in torch.arange(0, max_iterations):
        # Training variables defined as: loss, mse, regularisation
        
        
        running_loss = 0.0
        running_loss_NN = 0.0
        running_loss_Coeff = 0.0
        
        
        
        for batch_idx, train_sample in enumerate(train_dataloader):
            
            
            data_train, target_train = train_sample
            
            
            # ================== Training Model ============================
            
            
            #print("we are here at line 60 training")
            #print(data_train)
            #print("*******************************")
            
            
            # data_tain is ''time''
            'A linear NN considered to estimate the coefficients'
            
            
            prediction, time_derivs, thetas, output_nn = model(data_train)
            
            
            
            Num = thetas @ torch.t(model.estimated_coeffs.numerator.weight)
            
            
            
            Den = thetas[:,1:] @ torch.t(model.estimated_coeffs.denominator.weight)
            
            
            
            #####################################
            #####################################
            #####################################
            
            
            output_coeff_rational = Num/(Den + 1)
            
            
            #####################################
            #####################################
            #####################################
            
            
            
            
            
            #loss_3 = loss_fn(prediction,target_train)
            
            
            
            loss_3 = torch.mean((prediction - target_train)**2)
            
            
            loss_prediction_target = torch.mean((prediction - target_train)**2)
            
            
            
            "loss_6 : Definition of the loss function which is the subtraction of two terms:"
            "the first term is time_derivs and the second term is the estimated_coefficients of the new NN "
            
            
            
            #loss_6 = loss_fn(time_derivs, thetas @ torch.t(model.estimated_coeffs.numerator.weight))
            
            #output_coeff_rational
            
            loss_6 = loss_fn(time_derivs , output_coeff_rational)
            
            loss_time_coeff = loss_fn(time_derivs , output_coeff_rational)
            
            #print("7777777777777777777777777777")
            #print("7777777777777777777777777777")
            #print("7777777777777777777777777777")
            #vector_ones = torch.ones_like(time_derivs)
            #print(vector_ones.size())
            #print(Num.size())
            #print(time_derivs.size())
            #print(torch.mul(time_derivs , Num).size())
            #print("7777777777777777777777777777")
            #print("7777777777777777777777777777")
            #print("7777777777777777777777777777")

            
            #loss_6 = loss_fn( torch.mul(time_derivs , Den) , Num)
            
            
            "loss_7 : To have another defintiion for loss we considered the subtraction of"
            "time derivs and the output of new neural network"
            "this loss is the same as loss_6"
            
            
            
            loss_7 = loss_fn(time_derivs,output_nn)
            
            
            
            
            #print("77777777777777777777777777")
            #print("77777777777777777777777777")
            #print("77777777777777777777777777")
            #print(time_derivs.size())
            #print(thetas.size())
            #print(model.estimated_coeffs.linear.weight)
            #print("77777777777777777777777777")
            #print("77777777777777777777777777")
            #print("77777777777777777777777777")
            
            
            
            
            #loss_4 = loss_fn(time_derivs, thetas @ model.constraint_coeffs(scaled=False, sparse=True))
            
            
            "main loss which is the summation of two loss "
            loss_5 = 1 * loss_3 + 1e-1 * loss_6
            
            loss_5 = 1 * loss_prediction_target + 1e-1 * loss_time_coeff
            
                        
            # Optimizer step
            optimizer.zero_grad()
                        
            
            loss_5.backward()
            
            
                        
            optimizer.step()
            
            
            running_loss += loss_5.item()
            running_loss_Coeff += loss_6.item()
            running_loss_NN += loss_3.item()
        
        "tracking the loss function within the training loop"
        loss_values.append(running_loss)
        loss_values_NN.append(running_loss_NN)
        loss_values_Coeff.append(running_loss_Coeff)
            
            
                                
        loss = loss_5.cpu().detach()
        
       
       #device=dataset.device_type() 
       
        
        if iteration % write_iterations == 0:
            # ================== Validation costs ================
            #with torch.no_grad():
            #    batch_mse_test = torch.zeros(
            #        (n_features, len(test_dataloader)), device = device
            #    )
            #    for batch_idx, test_sample in enumerate(test_dataloader):
            #        data_test, target_test = test_sample
            #        prediction_test = model.func_approx(data_test)[0]
            #        batch_mse_test[:, batch_idx] = torch.mean(
            #            (prediction_test - target_test) ** 2, dim=-2
            #        )  # loss per output
            #mse_test = batch_mse_test.cpu().detach().mean(dim=-1)
            
            
            #print(estimated_coeffs.linear.weight)
            

            # ================== Sparsity update =============
            #Updating sparsity
            #update_sparsity = sparsity_scheduler(
            #    iteration, torch.sum(mse_test), model, optimizer
            #)
            
            
            if iteration > 30000 and iteration % write_iterations == 0:
                
                
                
                
                Ws = model.estimated_coeffs.numerator.weight.detach().clone()
                
                #Ws = model.estimated_coeffs.linear.weight
                
                
                
                
                Ws = model.estimated_coeffs.numerator.weight.detach().clone()
                Wd = model.estimated_coeffs.denominator.weight.detach().clone()
                
                Mask_Ws = (Ws.abs() > 0.2 ).type(torch.float)
                Mask_Wd = (Wd.abs() > 0.02 ).type(torch.float)
                
                model.estimated_coeffs.numerator.weight = torch.nn.Parameter(Ws * Mask_Ws)
                model.estimated_coeffs.denominator.weight = torch.nn.Parameter(Wd * Mask_Wd)

                model.estimated_coeffs.numerator.weight.register_hook(lambda grad: grad.mul_(Mask_Ws))
                model.estimated_coeffs.denominator.weight.register_hook(lambda grad: grad.mul_(Mask_Wd))
                
                
                optimizer = torch.optim.Adam([
                                {'params': model.func_approx.parameters(), 'lr': 5e-4/lr_red, 'weight_decay': 1e-4},
                                {'params': model.estimated_coeffs.parameters(), 'lr': 1e-3/lr_red, 'weight_decay': 0.0},
          #                      {'params': model.library.parameters(), 'lr': 5e-2, 'weight_decay': 1e-4},
                            ])
                
                
                lr_red = 1.5*lr_red
                
                #print(len(Ws[0]))
                
            #   print("99999999999999999999999999999")
                
                
                #Mask_Ws = (Ws.abs() > 0.2 ).type(torch.float)
                
                
                #print("99999999999999999999999999999")
                
                #print(Mask_Ws)
                #lambda grad: grad.mul_(Mask_Ws)
                #print(torch.nn.Parameter(Ws * Mask_Ws))
                
                #print(Ws.mul_(Mask_Ws))
                
                #print(lambda grad: grad * 2)
                
                #model.estimated_coeffs.linear.weight = torch.nn.Parameter(Ws * Mask_Ws)
                
                #model.estimated_coeffs.linear.weight.register_hook(lambda grad: grad.mul_(Mask_Ws)) 
                
                print("The weights during the training are:")
                
                #print(model.estimated_coeffs.linear.weight.grad)
                
                print(model.estimated_coeffs.numerator.weight)
                print(model.estimated_coeffs.denominator.weight)
                
                #print("999999999999999999999999999999")
                
                #print(model.constraint_coeffs(scaled=False, sparse=True))
                
                #model.constraint.sparsity_masks = model.sparse_estimator(
                #    thetas, time_derivs)
                    
            #    print("=========")
            #    print("=========")
            #    print("@Iteration:",iteration)
                
                #print(iteration)
                
                #print("Sparsity masks are:",model.constraint.sparsity_masks)
            #    print("=========")
            #    print("=========")

    return loss_values, loss_values_NN, loss_values_Coeff



                    
            
           
           
