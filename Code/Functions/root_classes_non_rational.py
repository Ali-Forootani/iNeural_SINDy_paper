#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:32:14 2022

@author: forootani
"""

""" 
    Some of these classes are borrowed from from Deepymod see: https://phimal.github.io/DeePyMoD/
    I) The constraint class that constrains the neural network with the obtained solution,
    II) The sparsity estimator class,
    III) Function library class on which the model discovery is performed.
    IV) The complete_network class integrates these seperate building blocks.
    These are all abstract classes and implement the flow logic, rather than the specifics.
"""

from typing import Tuple, List, NewType
import numpy as np
import torch.nn as nn
import torch
from abc import ABCMeta, abstractmethod


class Constraint(nn.Module, metaclass = ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the constraint module.
            Borrowed from Deepymod see: https://phimal.github.io/DeePyMoD/
        """
        super().__init__()
        self.sparsity_masks = None      
    def forward(self, input):
        time_derivs, thetas = input
        if self.sparsity_masks is None:
            self.sparsity_masks = torch.ones(thetas.shape[1], dtype=torch.bool).to(thetas.device)
        sparse_thetas, masks = self.apply_mask(thetas, self.sparsity_masks)
        coeff_vectors = self.fit(sparse_thetas, time_derivs)
        mapped_coeffs = self.map_coeffs(masks, coeff_vectors)
        if masks.shape[0] != coeff_vectors.shape[0]:
            self.coeff_vectors = self.map_coeffs(masks, coeff_vectors)
        else:
            self.coeff_vectors = coeff_vectors * masks[:, None] 
        return self.coeff_vectors
    @staticmethod
    def apply_mask(thetas, masks):
        sparse_thetas = thetas[:,masks]
        return sparse_thetas, masks
    @staticmethod
    def map_coeffs(mask: torch.Tensor, coeff_vector: torch.Tensor) -> torch.Tensor:
        mapped_coeffs = (
            torch.zeros((mask.shape[0], 1))
            .to(coeff_vector.device)
            .masked_scatter_(mask[:, None], coeff_vector)
        )
        return mapped_coeffs
    @abstractmethod
    def fit(self, sparse_thetas, time_derivs):
        raise NotImplementedError


class Estimator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Abstract baseclass for the sparse estimator module.
            Borrowed from Deepymod see: https://phimal.github.io/DeePyMoD/
            It can be used for further development
        """
        super().__init__()
        self.coeff_vectors = None
    def forward(self, thetas, time_derivs):
        # we first normalize theta and the time deriv
        with torch.no_grad():
            normed_time_derivs = (time_derivs / torch.norm(time_derivs)).detach().cpu().numpy()
            normed_thetas = (thetas / torch.norm(thetas, dim=0, keepdim=True)).detach().cpu().numpy()
        self.coeff_vectors = self.fit(thetas, time_derivs.squeeze())[:, None]
        sparsity_masks = torch.tensor(self.coeff_vectors != 0.0, dtype=torch.bool).squeeze().to(thetas[0].device) 
        return sparsity_masks

class Library(nn.Module):
    def __init__(self) -> None:
        """Abstract base class for the library module.
            The library module inheriets from this class.
        """
        super().__init__()
        self.norms = None
    def forward(self, input):
        """
        Parent class to creat liberay objects and
        Args:
            input: Tuple[torch.Tensor, torch.Tensor] --> [prediction, data] 
        """
        time_derivs, thetas = self.library(input)
        # In case to need normalization of the library we can consider the following
        self.norms = (torch.norm(time_derivs) / torch.norm(thetas, dim=0, keepdim=True)).detach().squeeze()
        return time_derivs , thetas

## Define coefficients for a dictionary
class CoeffsNetwork(nn.Module):
    def __init__(self, n_combinations, n_features):
        '''
        Defining the sparse coefficiets and in the forward pass, 
        we obtain multiplication of features and sparse coefficients.
        ----------
        n_combinations : int: the number of features in dictionary
            DESCRIPTION.
        n_features : int : the number of variables
            DESCRIPTION.

        Returns
        -------
        Product of features multiplied by sparse coefficients.
        '''
        super().__init__()
        self.linear = nn.Linear(n_combinations,n_features,bias=False)
        # Setting the weights to zeros
        #self.linear.weight = torch.nn.Parameter(1 * self.linear.weight.clone().detach())
        
    def forward(self,x):
        return self.linear(x)

class CompleteNetwork(nn.Module):
    """ The class that manges the entire NN networks
        Args:
            function_approximator: NN sturcture that we use to approximate our dynamic systems, t -> x
            library: library that is defined in the other module and is used to construct our library
            estimated_coeffs: single layer nn.module that is used in algorithm, e.g. 'time_derive = estimated_coeffs * library'

        for the case of rational function we can put more coefficients terms here!
    """

    def __init__(
        self,
        function_approximator: torch.nn.Sequential,
        #library: Library,
        library_k: Library,
        #estimated_coeffs: torch.nn.Sequential,
        estimated_coeffs_k: torch.nn.Sequential,
    ) -> None:
        
        super().__init__()
        self.func_approx = function_approximator
        #self.library = library
        self.library_k = library_k
        #self.estimated_coeffs = estimated_coeffs
        self.estimated_coeffs_k = estimated_coeffs_k
        
    def forward(self, input):
        
        prediction, coordinates = self.func_approx(input.float())

        #time_derivs, thetas = self.library((prediction, coordinates))
        #output_nn = self.estimated_coeffs(thetas)
        

        time_derivs_k, thetas_k = self.library_k((prediction, coordinates))
        output_nn_k = self.estimated_coeffs_k(thetas_k)
        
        return (prediction,
                #time_derivs, thetas, output_nn,
                time_derivs_k, thetas_k, output_nn_k)
    
    @property
    def sparsity_masks(self):
        """Returns the sparsity masks which contain the active terms.
            Borrowed from Deepymod see https://phimal.github.io/DeePyMoD/
            For fruther development can be used
        """
        return self.constraint.sparsity_masks
    
    
    def estimator_coeffs(self):
        """Calculate the coefficients as estimated by the sparse estimator.
            it is not used in our algoritm, borrowed from Deepymod, see: https://phimal.github.io/DeePyMoD/
        Returns:
            (TensorList): List of coefficients of size [(n_features, 1) x n_outputs]
        """
        coeff_vectors = self.sparse_estimator.coeff_vectors
        return coeff_vectors

    def constraint_coeffs(self, scaled, sparse):
        coeff_vectors = self.constraint.coeff_vectors
        if scaled:
            coeff_vectors = torch.div( coeff_vectors, self.library.norms)
        if sparse:
            coeff_vectors = coeff_vectors * self.sparsity_masks[:, None]   
        return coeff_vectors
