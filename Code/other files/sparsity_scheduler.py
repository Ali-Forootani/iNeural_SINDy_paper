""" Contains classes that schedule when the sparsity mask should be applied """
import torch
import numpy as np


class Periodic:
    

    def __init__(self, periodicity, initial_iteration):
        
        self.periodicity = periodicity
        self.initial_iteration = initial_iteration

    def __call__(self, iteration, loss, model, optimizer):
        # Update periodically
        apply_sparsity = False  # we overwrite it if we need to update

        if (iteration - self.initial_iteration) % self.periodicity == 0:
            
            apply_sparsity = True

        return apply_sparsity

