#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:55:03 2022
@author: forootani
"""

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from scipy.integrate import odeint, ode, solve_ivp
from sympy import *
import matplotlib.pyplot as plt
import random

#####################################
#####################################
#####################################

from abc import ABCMeta, abstractstaticmethod
from scipy.integrate import odeint, ode, solve_ivp
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Functions.utiles import NormalizedData, time_scaling_func


########################################
########################################
########################################
########################################
########################################
########################################
########################################


class Interface_Dynamic_System(metaclass = ABCMeta):

    @abstractstaticmethod
    def Diff_Equation(self):
        pass



class Lorenz(Interface_Dynamic_System):
        
    @staticmethod   
    def Diff_Equation(state,t):
        #lorenz system
        #print(initial_condition)
        x, y, z = state  # Unpack the state vector
                
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

class Two_D_Oscillator(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        
        x, y = state
        
        return -0.1 * x + 2 * y + 0, -2 * x - 0.1 * y + 0


class First_Order_Dyn_Sys(Interface_Dynamic_System):
    @staticmethod
    def Diff_Equation(state,t):
        
        x = state
        
        return -2 * x

class Three_D_Non_Linear(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        
        x,y,z = state  # Unpack the state vector
        return y,  z, -3*x + -1*y + -2.67*z + -1*x*z 
    
    



class Fitz_Hugh_Nagumo(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        v, w = state
        a_1:float = -0.33333
        a_2:float = 0.04
        a_3:float = -0.028
        b_1:float = 0.5
        b_2:float = 0.032
        
        return v - w  -0.33333 * v*v*v + 0.5 , 0.04 * v - 0.028 * w + 0.032




class Fitz_Hugh_Nagumo_2(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(state,t):
        v, w = state
        a_1:float = -0.33333
        a_2:float = 0.04
        a_3:float = -0.028
        b_1:float = 0.5
        b_2:float = 0.032
        
        return v - w  -0.33333 * v*v*v + 0.1 , 0.1 * v - 0.1 * w




class Cubic_Damped_Oscillator(Interface_Dynamic_System):
    @staticmethod
    def Diff_Equation(state,t):
        x,y = state        
        return -0.1*x**3 + 2* y**3, -2*x**3 - 0.1* y**3 


class GlycolyticOsci_model(Interface_Dynamic_System):
    
    @staticmethod
    def Diff_Equation(x,t):
        k1,k2,k3,k4,k5,k6 = (100.0, 6.0, 16.0, 100.0, 1.28, 12.0)
        j0,k,kappa,q,K1, phi, N, A = (2.5,1.8, 13.0, 4.0, 0.52, 0.1, 1.0, 4.0)
        ##
        dx0 = j0 - (k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q)

        dx1 = (2*k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q) - k2*x[1]*(N-x[4]) - k6*x[1]*x[4]

        dx2 = k2*x[1]*(N-x[4]) - k3*x[2]*(A-x[5])

        dx3 = k3*x[2]*(A-x[5]) - k4*x[3]*x[4] - kappa*(x[3]- x[6])

        dx4 = k2*x[1]*(N-x[4]) - k4*x[3]*x[4] - k6*x[1]*x[4]

        dx5 = -(2*k1*x[0]*x[5])/(1.0 + (x[5]/K1)**q) + 2*k3*x[2]*(A-x[5]) - k5*x[5]

        dx6 = phi*kappa*(x[3]-x[6]) - k*x[6]

        return np.array([dx0,dx1,dx2,dx3,dx4,dx5,dx6])


##########################
##########################

class Factory_Dyn_Sys():
    
    def __init__(self,t,list_initial_conditions,
                 scaling_factor,func_name,add_noise,noise_level):
        
        self.t = t
        self.list_initial_conditions = list_initial_conditions
        self.scaling_factor = scaling_factor
        self.func_name = func_name
        self.add_noise = add_noise
        self.noise_level = noise_level
    
    
    def function_data(self):
        
        try:
            
            if self.func_name == "lorenz":
                dyn_sys_obj = Lorenz()
                
                
            if self.func_name == "2_D_Oscilator":
                
                dyn_sys_obj = Two_D_Oscillator()
                
                
            if self.func_name == "Fitz-Hugh Nagumo":
                
                #dyn_sys_obj = Fitz_Hugh_Nagumo()
                dyn_sys_obj = Fitz_Hugh_Nagumo_2()
                
            if self.func_name == "Cubic Damped Oscillator":
                
                dyn_sys_obj = Cubic_Damped_Oscillator()
                
            if self.func_name == "GlycolyticOsci_model":
            
                dyn_sys_obj = GlycolyticOsci_model()
            if self.func_name == "First Order Dynamic System":
                dyn_sys_obj = First_Order_Dyn_Sys()
                
            if self.func_name == "3_D_Non_Linear":
                
                dyn_sys_obj = Three_D_Non_Linear()
                
            return dyn_sys_obj    
            
            
            
            
            raise AssertionError("Please check the list of dynamical systems or insert your custom one")
            
            #u = odeint(dyn_sys_obj.Diff_Equation, [2,3], t)
            #print(dyn_sys_obj)
            
            
            
            
        except AssertionError as _e:
                print(_e)
                
    def data_prepration(self, dyn_sys_obj):
        
        num_features = len(self.list_initial_conditions[0])
        u_original_main = []
        initial_cond_main = []
        t_scaled_main = []
        true_data_noise_free = []
        
        
        
        second_list = torch.tensor(self.list_initial_conditions)
        
        second_list = second_list / second_list.abs().max(0,keepdim=True)[0]
        
        
        
        
        for i in range(0, len(self.list_initial_conditions)):
            
            initial_condition = second_list[i,:]
            
            
            u_original = odeint(dyn_sys_obj.Diff_Equation, self.list_initial_conditions[i], self.t)
            
            true_data_noise_free.append(u_original) 
            
            
            if self.add_noise == True:
                f_noise = lambda mean,std,num_samples: np.random.normal(mean, std, size=num_samples)
                noise_val = f_noise(0,1, int(self.t.size()[0]))
                u_original = np.add(u_original, self.noise_level * noise_val.reshape(-1,1)) 
            
            
            
            u_original_main.append(torch.from_numpy(self.scaling_factor*u_original))
            
            
            
            vector_initial_cond = torch.mul(initial_condition
                                            ,torch.ones_like(torch.from_numpy(u_original)))
            
            
            
            
            initial_cond_main.append(vector_initial_cond)
            
            
            
            t_scaled_main.append(time_scaling_func(self.t))
            
        
        
        list_t_scaled_main = t_scaled_main
        
        list_u_original = u_original_main
        
        
        t_scaled_main = torch.stack(t_scaled_main, dim=0)
        
        u_original_main = torch.stack(u_original_main, dim=0)
        
        initial_cond_main = torch.stack(initial_cond_main, dim=0)
        
        list_initial_cond_main = initial_cond_main
        
        
        t_scaled_main = torch.reshape(t_scaled_main,
                                      (len(self.list_initial_conditions)*self.t.size()[0],1))
        
        u_original_main = torch.reshape(u_original_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))
        
        initial_cond_main = torch.reshape(initial_cond_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))

        
        return (t_scaled_main, initial_cond_main,
                u_original_main,
                list_t_scaled_main, list_u_original, list_initial_cond_main, true_data_noise_free)
        
    
    def run(self):
            
        dyn_sys_obj = self.function_data()
        
        (t_scaled_main, initial_cond_main,
                u_original_main,
                list_t_scaled_main,
                list_u_original, list_initial_cond_main, true_data_noise_free) = self.data_prepration(dyn_sys_obj)
        #u = odeint(dyn_sys_obj.Diff_Equation,self.list_initial_conditions[0],self.t)
            
        return (t_scaled_main, initial_cond_main,
                u_original_main,
                list_t_scaled_main, list_u_original, list_initial_cond_main, true_data_noise_free)
    





####################################
####################################
####################################

####################################
####################################
####################################

####################################
####################################




####################################
####################################
####################################

####################################
####################################
####################################

####################################
####################################







###################
###################
###################
    

###################
###################
################### Ashwin section

#import list_functions


class rational_A:
    
    def __init__(self, t : torch.tensor, 
                 list_initial_conditions, scaling_factor, function_name) :
        
        #self.func = getattr(self, function_name)
        self.func = list_functions.__dict__[foo]


#####################
