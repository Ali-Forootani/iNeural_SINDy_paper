#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:19:04 2022

@author: forootani
"""


import torch
import torch.nn as nn
from typing import List, Tuple
import numpy as np




class NN_network_tanh(nn.Module):
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int) -> None:
        
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        coordinates = input.clone().detach().requires_grad_(True)
        return self.network(coordinates), coordinates

    def build_network(
        self, n_in: int, n_hidden: List[int], n_out: int
    ) -> torch.nn.Sequential:
        

        network = []
        architecture = [n_in] + n_hidden + [n_out]
        for layer_i, layer_j in zip(architecture, architecture[1:]):
            network.append(nn.Linear(layer_i, layer_j))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        return nn.Sequential(*network)



class NN_network_sigmoid(nn.Module):
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int) -> None:
        
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        coordinates = input.clone().detach().requires_grad_(True)
        return self.network(coordinates), coordinates

    def build_network(
        self, n_in: int, n_hidden: List[int], n_out: int
    ) -> torch.nn.Sequential:
        

        network = []
        architecture = [n_in] + n_hidden + [n_out]
        for layer_i, layer_j in zip(architecture, architecture[1:]):
            network.append(nn.Linear(layer_i, layer_j))
            network.append(nn.Sigmoid())
        network.pop()  # get rid of last activation function
        return nn.Sequential(*network)


class NN_network_relu(nn.Module):
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int) -> None:
        
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        coordinates = input.clone().detach().requires_grad_(True)
        return self.network(coordinates), coordinates

    def build_network(
        self, n_in: int, n_hidden: List[int], n_out: int
    ) -> torch.nn.Sequential:
        

        network = []
        architecture = [n_in] + n_hidden + [n_out]
        for layer_i, layer_j in zip(architecture, architecture[1:]):
            network.append(nn.Linear(layer_i, layer_j))
            network.append(nn.ReLU())
        network.pop()  # get rid of last activation function
        return nn.Sequential(*network)





#class weightedTanh(nn.Module):
#    def __init__(self, weights = 1):
#        super().__init__()
#        self.weights = weights

#    def forward(self, input):
#        ex = torch.exp(2*self.weights*input)
#        return (ex-1)/(ex+1)




class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30,
        is_first: bool = False,
    ) -> None:
        """Sine activation function layer with omega_0 scaling.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            omega_0 (float, optional): Scaling factor of the Sine function. Defaults to 30.
            is_first (bool, optional): Defaults to False.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialization of the weigths."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            input (torch.Tensor): Input tensor of shape (n_samples, n_inputs).

        Returns:
            torch.Tensor: Prediction of shape (n_samples, n_outputs)
        """
        return torch.sin(self.omega_0 * self.linear(input.float()))





class Siren(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hidden: List[int],
        n_out: int,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
    ) -> None:
        """SIREN model from the paper [Implicit Neural Representations with
        Periodic Activation Functions](https://arxiv.org/abs/2006.09661).

        Args:
            n_in (int): Number of input features.
            n_hidden (list[int]): Number of neurons in each layer.
            n_out (int): Number of output features.
            first_omega_0 (float, optional): Scaling factor of the Sine function of the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): Scaling factor of the Sine function of the hidden layers. Defaults to 30.
        """
        super().__init__()
        self.network = self.build_network(
            n_in, n_hidden, n_out, first_omega_0, hidden_omega_0
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        coordinates = input.clone().detach().requires_grad_(True)
        return self.network(coordinates), coordinates

    def build_network(
        self,
        n_in: int,
        n_hidden: List[int],
        n_out: int,
        first_omega_0: float,
        hidden_omega_0: float,
    ) -> torch.nn.Sequential:
        """Constructs the Siren neural network.

        Args:
            n_in (int): Number of input features.
            n_hidden (list[int]): Number of neurons in each layer.
            n_out (int): Number of output features.
            first_omega_0 (float, optional): Scaling factor of the Sine function of the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): Scaling factor of the Sine function of the hidden layers. Defaults to 30.
        Returns:
            torch.Sequential: Pytorch module
        """
        network = []
        # Input layer
        network.append(
            SineLayer(n_in, n_hidden[0], is_first=True, omega_0=first_omega_0)
        )

        # Hidden layers
        for layer_i, layer_j in zip(n_hidden, n_hidden[1:]):
            network.append(
                SineLayer(layer_i, layer_j, is_first=False, omega_0=hidden_omega_0)
            )

        # Output layer
        final_linear = nn.Linear(n_hidden[-1], n_out)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6 / n_hidden[-1]) / hidden_omega_0,
                np.sqrt(6 / n_hidden[-1]) / hidden_omega_0,
            )
            network.append(final_linear)

        return nn.Sequential(*network)

 
## Residual blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_features, activation = nn.ELU):
        super(ResidualBlock, self).__init__()
        self.activation = activation
        
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            self.activation(),
            nn.Linear(in_features, in_features),
        )
        
    def forward(self,x):
        # return self.block(x)
        return x + self.block(x)


   
## ResNet for nonlinear part 
class ODE_Net(nn.Module):
    def __init__(self,n,num_residual_blocks, hidden_features = 25, activation = nn.ELU, print_model = True):
        super(ODE_Net,self).__init__()
        self.activation = activation
        model = [
            nn.Linear(n, hidden_features),
            ]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(hidden_features, activation = self.activation)]
            
        model += [
            nn.Linear(hidden_features,n),
            ]        
        
        # model = [nn.Linear(n,n),]
        self.model = nn.Sequential(*model)
        
        if print_model:
            print(self.model)
        
    def forward(self,x):
        return self.model(x)


