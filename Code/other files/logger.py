""" Module to log performance metrics whilst training Deepmyod """
import numpy as np
import torch
import sys, time
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, exp_ID, log_dir):
        """Log the training process of Deepymod.
        Args:
            exp_ID (str): name or ID of the this experiment
            log_dir (str): directory to save the log files to disk.

        """
        self.writer = SummaryWriter(
            comment=exp_ID, log_dir=log_dir, max_queue=5, flush_secs=10
        )
        self.log_dir = self.writer.get_logdir()

    def __call__(
        self,
        iteration,
        running_loss,
        loss_values_Coeff,
        loss_values_NN,
        loss_values_RK4,
        estimated_coeffs_k,
        estimated_coeffs_numerator,
        estimated_coeffs_denominator,
        write_iterations,
        threshold_iteration,
        **kwargs,
    ):  
        #l1_norm = torch.sum(torch.abs(torch.cat(constraint_coeffs, dim=1)), dim=0)
        
        
        l1_norm = torch.norm(estimated_coeffs_k,1)
        
        
        
        #iteration,
        #loss_values,
        #loss_values_Coeff,
        #loss_values_NN,
        #loss_values_RK4,
        #model.estimated_coeffs_k.linear.weight,
        #model.estimated_coeffs.numerator.weight,
        #model.estimated_coeffs.denominator.weight,
        
        
        
        #self.update_tensorboard(
        #    iteration,
        #    running_loss,
        #    MSE,
        #    Reg,
        #    l1_norm,
        #    constraint_coeffs,
        #    unscaled_constraint_coeffs,
        #    estimator_coeffs,
        #    **kwargs,
        #)
        self.update_terminal(iteration, loss_values_Coeff,
                             loss_values_NN, loss_values_RK4,running_loss)
        
        self.update_coefficients(estimated_coeffs_k,
                            estimated_coeffs_numerator,
                            estimated_coeffs_denominator,
                            iteration,
                            write_iterations,
                            threshold_iteration
                            )
        
        #constraint_coeff_vectors,
        #unscaled_constraint_coeff_vectors,
        #estimator_coeff_vectors, iteration, write_iterations,
        #threshold_iteration
        
    def update_tensorboard(
        self,
        iteration,
        running_loss,
        loss_values_Coeff,
        loss_values_NN,
        loss_values_RK4,
        estimated_coeffs_k,
        estimated_coeffs_numerator,
        estimated_coeffs_denominator,
        write_iterations,
        threshold_iteration,
        **kwargs,
    ):
        """Write the current state of training to Tensorboard
        Args:
            
        """
        #print("========================")
        #print("========================")
        #print(loss_mse)
        
        
        #for idx, val in enumerate(loss_mse):
            
        #    print(val)
            
        #    pass
        
        
        
        # Costs and coeff vectors
        #self.writer.add_scalars("loss/loss", loss, iteration)
        
        
        
        
        self.writer.add_scalars(
            "loss/mse",
            {f"output_{idx}": val for idx, val in enumerate(running_loss)},
            iteration,
        )
        self.writer.add_scalars(
            "loss/reg",
            {f"output_{idx}": val for idx, val in enumerate(loss_values_Coeff)},
            iteration,
        )
        #self.writer.add_scalars(
        #    "loss/l1",
        #    {f"output_{idx}": val for idx, val in enumerate(loss_l1)},
        #    iteration,
        #)

        for output_idx, (coeffs_k, coeffs_num, coeffs_den) in enumerate(
            zip(
                estimated_coeffs_k,
                estimated_coeffs_numerator,
                estimated_coeffs_denominator,
            )
        ):
            self.writer.add_scalars(
                f"coeffs/output_{output_idx}",
                {f"coeff_{idx}": val for idx, val in enumerate(coeffs_k.squeeze())},
                iteration,
            )
            self.writer.add_scalars(
                f"coeffs_num/output_{output_idx}",
                {
                    f"coeff_{idx}": val
                    for idx, val in enumerate(coeffs_num.squeeze())
                },
                iteration,
            )
            self.writer.add_scalars(
                f"coeffs_den/output_{output_idx}",
                {
                    f"coeff_{idx}": val
                    for idx, val in enumerate(estimated_coeffs_denominator.squeeze())
                },
                iteration,
            )

        # Writing remaining kwargs
        for key, value in kwargs.items():
            if value.numel() == 1:
                self.writer.add_scalar(f"remaining/{key}", value, iteration)
            else:
                self.writer.add_scalars(
                    f"remaining/{key}",
                    {
                        f"val_{idx}": val.squeeze()
                        for idx, val in enumerate(value.squeeze())
                    },
                    iteration,
                )

    def update_terminal(self, iteration,
                        loss_values_Coeff, loss_values_NN,loss_values_RK4,running_loss):
        """Prints and updates progress of training cycle in command line."""
        sys.stdout.write(
            f"\r{iteration:>6} running_loss: {running_loss[-1]:>8.2e} \
            loss_values_Coeff: {loss_values_Coeff[-1]:>8.2e} \
                loss_values_NN: {loss_values_NN[-1]:>8.2e} \
                    loss_values_RK4: {loss_values_RK4[-1]:>8.2e} "
        )
        sys.stdout.flush()
        
        
    def update_coefficients(self, 
    estimated_coeffs_k,
    estimated_coeffs_numerator,
    estimated_coeffs_denominator,
    iteration, write_iterations,
    threshold_iteration):
        
        #model.estimated_coeffs_k.linear.weight,
        #model.estimated_coeffs.numerator.weight,
        
        #estimated_coeffs_k,
        #estimated_coeffs_numerator,
        #estimated_coeffs_denominator,
        #iteration,
        #write_iterations,
        #threshold_iteration
        
        # if iteration > threshold_iteration and iteration % write_iterations == 0:
        
        
        if iteration > threshold_iteration and iteration % write_iterations == 0:
            
            #print("888888888888888888888")
            #print("888888888888888888888")
            #print(estimated_coeffs_k)
            
            sys.stdout.write('\n')
            sys.stdout.write("=================")
            sys.stdout.write('\n')
            sys.stdout.write("=================")
            sys.stdout.write('\n')   
            sys.stdout.write(
                f"\r{iteration>0}  Coefficients_K:  \
                    {estimated_coeffs_k}"
            )
                
            sys.stdout.write('\n')    
            #sys.stdout.write(f"Coefficients_Num: \
            #    {estimated_coeffs_numerator}" )
            #sys.stdout.write('\n')
            #sys.stdout.write(f"Coefficients_Den: \
            #    {estimated_coeffs_denominator}" )
                
            sys.stdout.flush()
        
        

    def close(self, model):
        """Close the Tensorboard writer"""
        print("Algorithm converged. Writing model to disk.")
        self.writer.flush()  # flush remaining stuff to disk
        self.writer.close()  # close writer

        # Save model
        model_path = self.log_dir + "model.pt"
        torch.save(model.state_dict(), model_path)