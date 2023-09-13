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

from utiles import normalized_data, time_scaling_func

from sympy import *
import matplotlib.pyplot as plt



#####################################
#####################################
#####################################



from abc import ABCMeta, abstractstaticmethod
from scipy.integrate import odeint, ode, solve_ivp
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utiles import normalized_data, time_scaling_func



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
        
        return -0.1 * x + 2 * y + 0, -2 * x + 0.1 * y + 0


class First_Order_Dyn_Sys(Interface_Dynamic_System):
    @staticmethod
    def Diff_Equation(state,t):
        
        x = state
        
        return -20 * x 


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
    
    def __init__(self,t,list_initial_conditions,scaling_factor,func_name):
        
        self.t = t
        self.list_initial_conditions = list_initial_conditions
        self.scaling_factor = scaling_factor
        self.func_name = func_name
    
    
    def function_data(self):
        
        try:
            
            if self.func_name == "lorenz":
                dyn_sys_obj = Lorenz()
                
                
            if self.func_name == "2_D_Oscilator":
                
                dyn_sys_obj = Two_D_Oscillator()
                
                
            if self.func_name == "Fitz-Hugh Nagumo":
                
                dyn_sys_obj = Fitz_Hugh_Nagumo()
                
            if self.func_name == "Cubic Damped Oscillator":
                
                dyn_sys_obj = Cubic_Damped_Oscillator()
                
            if self.func_name == "GlycolyticOsci_model":
            
                dyn_sys_obj = GlycolyticOsci_model()
            if self.func_name == "First Order Dynamic System":
                dyn_sys_obj = First_Order_Dyn_Sys()
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
        
        
        for i in range(0, len(self.list_initial_conditions)):
            
            initial_condition = self.list_initial_conditions[i]
            
            u_original = odeint(dyn_sys_obj.Diff_Equation, initial_condition, self.t)
            
            u_original_main.append(torch.from_numpy(self.scaling_factor*u_original))
            
            
            
            vector_initial_cond = torch.mul(torch.tensor(initial_condition)
                                            ,torch.ones_like(torch.from_numpy(u_original)))
            vector_initial_cond = torch.div(vector_initial_cond,
                                            max(initial_condition, key=abs))
            
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
                list_t_scaled_main, list_u_original, list_initial_cond_main)
        
    
    def run(self):
            
        dyn_sys_obj = self.function_data()
        
        (t_scaled_main, initial_cond_main,
                u_original_main,
                list_t_scaled_main,
                list_u_original, list_initial_cond_main) = self.data_prepration(dyn_sys_obj)
        #u = odeint(dyn_sys_obj.Diff_Equation,self.list_initial_conditions[0],self.t)
            
        return (t_scaled_main, initial_cond_main,
                u_original_main,
                list_t_scaled_main, list_u_original, list_initial_cond_main)
    













####################################
####################################
####################################

####################################
####################################
####################################

####################################
####################################
####################################



def first_order_solution(time_constant: float, initial_condition: float, t: torch.tensor) -> torch.tensor:
    
    #k = torch.tensor(k)
    #gamma = torch.tensor(gamma)
    #m = torch.tensor(m)
    #MinMaxScaler()    
    #
    #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    #X_scaled = X_std * (max - min) + min
    #
    
    
    
    time_constant = torch.tensor(time_constant)
    initial_condition = torch.tensor(initial_condition)
    
    
        
    u = torch.mul( initial_condition, torch.exp(- time_constant * t))
    u_original = u
    
    
    
    #Scaling the output by a coefficient u.max
    
    u_std = torch.div(u ,torch.max(u) )
    u_scaled = torch.mul(u_std , 1 - 0) + 0
    
    
    
    
    
    
    ################
    ################
    
    
    
    # Mapping the time between [-1,1]
    
    t_std = torch.div(t - torch.min(t),torch.max(t) - torch.min(t) )
    
    t_scaled = torch.mul(t_std, 1 + 1) - 1

    
    #omega_n=torch.sqrt(k/m).view(-1,1)
    #xi=torch.div(gamma,2*omega_n*m).view(-1,1)
    #omega_d=torch.matmul(omega_n, torch.sqrt(1-xi))
    
    
    #coeficient = torch.div(xi,torch.sqrt(1-torch.pow(xi,2)))
    #u = 1 - torch.exp(- xi * omega_n * t)* torch.add(coeficient * torch.sin(omega_d *t), torch.cos(omega_d * t))
    
    coords = t_scaled.reshape(-1,1)
        
    return coords, u_scaled.view(-1,1), u_original


###################
###################
###################

#Parameter containing: for the cases of x^5
#tensor([[ 0.0000, -0.0000,  1.2561, -0.4141,  0.5225,  0.3570, -0.7495,  0.2231,
#          0.4601,  0.3753, -0.6828,  0.0000,  0.6583,  1.0515,  0.0000, -0.9408,
#         -0.0000, -0.0000,  0.0000,  0.0000, -0.0000],
#        [ 1.8206,  0.2190,  1.3181, -0.0000, -0.3160,  0.5362, -0.0000, -1.2408,
#         -0.4786, -0.0000,  0.0000, -1.4367, -0.7188, -0.0000, -0.2863,  0.0000,
#         -1.7891, -0.5043, -0.0000,  0.0000,  0.0000]], requires_grad=True)
#Denominator============================
#Parameter containing:
#tensor([[ 4.4203,  5.8651,  5.1781,  5.7490,  4.9516,  4.7164,  5.5349,  4.1939,
#          2.1738,  3.5967,  4.6210,  3.3717,  1.2295,  0.0000,  1.5684,  3.3158,
#          2.4324,  0.4432, -0.0000, -0.0000],
#        [ 1.2607,  1.6906,  0.7529,  0.3869,  0.4754,  1.9975,  1.3369,  0.5891,
#          0.4763,  3.1327,  2.2037,  1.0009,  0.0000, -0.0000,  2.1157,  2.1796,
#          0.9300, -0.0000, -0.3354,  0.2589]], requires_grad=True)


###################
###################
###################



@dataclass
class rational_2D:
    t: torch.tensor
    list_initial_conditions: list
    scaling_factor: float = 1
    a_1: float = 0
    a_2: float = 0
    a_3: float = 1.2561
    a_4: float = -0.4141
    a_5: float = 0.5225
    a_6: float = 0.357
    a_7: float = -0.7495
    a_8: float = 0.2231
    a_9: float = 0.4601
    a_10: float = 0.3753
    a_11: float = -0.6828
    a_12: float = 0
    a_13: float = 0.6583
    a_14: float = 1.0515
    a_15: float = 0
    a_16: float = -0.9408
    a_17: float = 0
    a_18: float = 0
    a_19: float = 0
    a_20: float = 0
    a_21: float = 0
    
    ###################
    ###################
    #tensor([[ 4.4203,  5.8651,  5.1781,  5.7490,  4.9516,  4.7164,  5.5349,  4.1939,
    #          2.1738,  3.5967,  4.6210,  3.3717,  1.2295,  0.0000,  1.5684,  3.3158,
    #          2.4324,  0.4432, -0.0000, -0.0000],
    b_1: float = 1
    b_2: float = 4.4203
    b_3: float = 5.8651
    b_4: float = 5.1781
    b_5: float = 5.7490
    b_6: float = 4.9516
    b_7: float = 4.7164
    b_8: float = 5.5349
    b_9: float = 4.1939
    b_10: float = 2.1738
    b_11: float = 3.5967
    b_12: float = 4.6210
    b_13: float = 3.3717
    b_14: float = 1.2295
    b_15: float = 0
    b_16: float = 1.5684
    b_17: float = 3.3158
    b_18: float = 2.34324
    b_19: float = 0.4432
    b_20: float = 0
    b_21: float = 0

    ###################
    ###################
    #        [ 1.8206,  0.2190,  1.3181, -0.0000, -0.3160,  0.5362, -0.0000, -1.2408,
    #         -0.4786, -0.0000,  0.0000, -1.4367, -0.7188, -0.0000, -0.2863,  0.0000,
    #         -1.7891, -0.5043, -0.0000,  0.0000,  0.0000]], requires_grad=True)
    c_1: float = 1.8206
    c_2: float = 0.2190
    c_3: float = 1.3181
    c_4: float = 0
    c_5: float = -0.3160
    c_6: float = 0.5362
    c_7: float = 0
    c_8: float = -1.2408
    c_9: float = -0.4786
    c_10: float = 0
    c_11: float = 0
    c_12: float = -1.4367
    c_13: float = -0.7188
    c_14: float = 0
    c_15: float = -0.2863
    c_16: float = 0
    c_17: float = -1.7891
    c_18: float = -0.5043
    c_19: float = 0
    c_20: float = 0
    c_21: float = 0
    
    ###################
    ###################
    
    #        [ 1.2607,  1.6906,  0.7529,  0.3869,  0.4754,  1.9975,  1.3369,  0.5891,
    #          0.4763,  3.1327,  2.2037,  1.0009,  0.0000, -0.0000,  2.1157,  2.1796,
    #          0.9300, -0.0000, -0.3354,  0.2589]], requires_grad=True)
    d_1: float = 1
    d_2: float = 1.2607
    d_3: float = 1.6906
    d_4: float = 0.7529
    d_5: float = 0.3869
    d_6: float = 0.4754
    d_7: float = 1.9975
    d_8: float = 1.3369
    d_9: float = 0.5891
    d_10: float = 0.4763
    d_11: float = 3.1327
    d_12: float = 2.2037
    d_13: float = 1.0009
    d_14: float = 0
    d_15: float = 0
    d_16: float = 2.1157
    d_17: float = 2.1796
    d_18: float = 0.9300
    d_19: float = 0
    d_20: float = -0.3354
    d_21: float = 0.2589
    
    
    
    #vs = 0.76 (μMh−1) k s = 0.38 (h−1)
    #vm = 0.65 (μMh−1) k1 = 1.9 (h−1)
    #vd = 0.95 (μMh−1) k2 = 1.3 (h−1)
    #V1 = 3.2 (μMh−1) K d = 0.2 (μM)
    #V2 = 1.58 (μMh−1) K I = 1 (μM)
    #V3 = 5 (μMh−1) K m = 0.5 (μM)
    #V4 = 2.5 (μMh−1) K1,2,3,4 = 2 (μM)
    #################
    a_1: float = 0.1
    a_2: float = 0.1
    a_3: float = 0.1
    b_1: float = 0.82
    b_2: float = 3
    
    #################
    #k_i: float = 1
    #k_d: float = 0.2
    #k_2: float = 1.3
    #k_1: float = 1.9
    #k_s: float = 0.38
    #k_m: float = 0.5
    #V1: float = 3.2
    #V2: float = 1.58
    #V3: float = 5
    #V4: float = 2.5
    #vm: float = 0.65
    #vd: float = 0.95
    #vs: float = 0.76
    
    ##################
    ##################
    
    #Numerator============================
    #Parameter containing:
    #tensor([[ 0.6419, -1.7686,  0.3720, -2.3068, -1.0339,  0.5113, -1.7315, -1.0898,
    #         -0.6535,  0.6400, -1.0593, -0.6928, -0.8323, -0.5797,  0.4593, -0.5439,
    #         -0.4236, -0.4350, -0.6840, -0.6674,  0.2314,  0.0000,  0.0000,  0.3403,
    #          0.0000, -0.2047, -0.6164,  0.2071],
    #        [ 1.0913,  1.4041,  0.8924,  0.0000,  0.7627,  0.8998,  0.0000, -0.0000,
    #          0.4918,  0.6791, -0.0000, -0.6707, -0.2628,  0.0000,  0.0000,  0.0000,
    #         -1.3805, -0.9715, -0.4404, -0.2905,  0.0000, -0.0000, -3.1844, -2.0843,
    #         -0.9508, -0.5780, -0.0000,  0.0000]], requires_grad=True)
    #Denominator============================
    #Parameter containing:
    #tensor([[ 7.9142,  8.6108,  6.4420,  7.3957,  7.0730,  4.2285,  5.3872,  5.2954,
    #          5.5050,  2.1428,  3.0398,  3.2394,  3.0561,  3.8713,  0.7253,  0.9664,
    #          1.3150,  1.3108,  1.4628,  2.3857,  0.0000, -0.0000,  0.0000,  0.0000,
    #         -0.0000,  0.0000,  1.3826],
    #        [ 2.3566,  4.8932, -2.4195,  0.0000,  2.7663, -1.7639, -1.3326, -2.0735,
    #          2.4340,  2.9003,  2.4481, -2.1664, -2.3395,  1.7161,  5.5047,  5.9845,
    #          2.2588, -1.4272, -1.7946,  0.2765,  3.1754,  3.0999,  3.8608,  3.9005,
    #          1.7216, -0.5702, -0.0000]], requires_grad=True)
    #=============
    
    #[ 0.6419, -1.7686,  0.3720, -2.3068, -1.0339,  0.5113, -1.7315, -1.0898,
    #         -0.6535,  0.6400, -1.0593, -0.6928, -0.8323, -0.5797,  0.4593, -0.5439,
    #         -0.4236, -0.4350, -0.6840, -0.6674,  0.2314,  0.0000,  0.0000,  0.3403,
    #          0.0000, -0.2047, -0.6164,  0.2071]
    
    #######################################
    #######################################
    ########## 
    #Numerator============================
    #Parameter containing:
    #tensor([[ 0.3758, -0.9865,  0.2694, -1.4437, -0.5623,  0.2964, -1.3887, -0.7559,
    #         -0.4128,  0.4362, -1.1767, -0.5391, -0.5715, -0.0000,  0.0000, -0.9952,
    #         -0.0000, -0.0000, -0.0000, -0.0000,  0.0000, -0.9360,  0.4568,  0.3555,
    #          0.0000, -0.0000,  0.0000,  0.0000],
    #        [ 1.3753,  1.2447,  1.0474,  0.5495,  0.4873,  0.7254, -0.0000, -0.4877,
    #          0.0000,  0.0000, -0.7432, -1.4123, -0.5496, -0.0000,  0.0000, -0.4169,
    #         -1.7772, -0.9935, -0.3791, -0.0000,  0.0000,  0.0000, -1.1870, -0.6619,
    #         -0.2097, -0.0000, -0.0000,  0.0000]], requires_grad=True)
    #Denominator============================
    #Parameter containing:
    #tensor([[ 4.8536,  4.3058,  5.6060,  4.4214,  3.7572,  5.3920,  3.8621,  3.2714,
    #          2.1892,  4.4685,  2.8593,  2.3248,  1.6062,  0.5754,  2.6374,  1.3860,
    #          0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.5777, -0.2188, -0.0000,
    #         -0.2647, -0.4124, -0.0000],
    #        [ 1.8057,  3.1958,  0.0000,  0.0000,  2.4179,  0.0000,  0.0000,  0.0000,
    #          0.4797,  1.3243,  0.0000, -0.0000,  0.0000,  0.0000,  5.8689,  0.3974,
    #         -0.0000,  0.0000,  0.0000,  0.0000,  4.5259,  2.9360,  2.4411,  0.0000,
    #          0.0000, -0.0000, -0.0000]], requires_grad=True)
    #=============
    
    
    
    
    a_1: float = 0.375
    a_2: float = -0.9865
    a_3: float = 0.2694
    a_4: float = -1.4437
    a_5: float = -0.5623
    a_6: float = 0.2964
    a_7: float = -1.3887
    a_8: float = -0.7559
    a_9: float = -0.4128
    a_10: float = 0.4362
    a_11: float = -1.1767
    a_12: float = -0.5391
    a_13: float = -0.5715
    a_14: float = 0
    a_15: float = 0
    a_16: float = -0.9952
    a_17: float = 0
    a_18: float = 0
    a_19: float = 0
    a_20: float = 0
    a_21: float = 0
    a_22: float = -0.9360
    a_23: float = 0.4568
    a_24: float = 0.3555
    a_25: float = 0
    a_26: float = 0
    a_27: float = 0
    a_28: float = 0
    
    ############
    #first try
    #[ 7.9142,  8.6108,  6.4420,  7.3957,  7.0730,  4.2285,  5.3872,  5.2954,
    #          5.5050,  2.1428,  3.0398,  3.2394,  3.0561,  3.8713,  0.7253,  0.9664,
    #          1.3150,  1.3108,  1.4628,  2.3857,  0.0000, -0.0000,  0.0000,  0.0000,
    #         -0.0000,  0.0000,  1.3826]
    
    #second try
    #[ 4.8536,  4.3058,  5.6060,  4.4214,  3.7572,  5.3920,  3.8621,  3.2714,
    #          2.1892,  4.4685,  2.8593,  2.3248,  1.6062,  0.5754,  2.6374,  1.3860,
    #          0.0000,  0.0000,  0.0000, -0.0000, -0.0000, -0.5777, -0.2188, -0.0000,
    #         -0.2647, -0.4124, -0.0000]
    
    
    
    b_1: float = 1
    b_2: float = 4.3058
    b_3: float = 4.3058
    b_4: float = 5.6060
    b_5: float = 4.4214
    b_6: float = 3.7572
    b_7: float = 5.3920
    b_8: float = 3.8621
    b_9: float = 3.2714
    b_10: float = 2.1892 
    b_11: float = 4.4685
    b_12: float = 2.8593
    b_13: float = 2.3248
    b_14: float = 1.6062
    b_15: float = 0.5754
    b_16: float = 2.6374
    b_17: float = 1.3860
    b_18: float = 0
    b_19: float = 0
    b_20: float = 0
    b_21: float = 0
    b_22: float = 0
    b_23: float = -0.5777
    b_24: float = -0.2188
    b_25: float = 0
    b_26: float = -0.2647
    b_27: float = -0.4124
    b_28: float = 0
    #b_27: float = 0.1
    #b_28: float = 0.1
    
    #######
    #first try
    #        [ 1.0913,  1.4041,  0.8924,  0.0000,  0.7627,  0.8998,  0.0000, -0.0000,
    #          0.4918,  0.6791, -0.0000, -0.6707, -0.2628,  0.0000,  0.0000,  0.0000,
    #         -1.3805, -0.9715, -0.4404, -0.2905,  0.0000, -0.0000, -3.1844, -2.0843,
    #         -0.9508, -0.5780, -0.0000,  0.0000]
    
    #second try
    #        [ 1.3753,  1.2447,  1.0474,  0.5495,  0.4873,  0.7254, -0.0000, -0.4877,
    #          0.0000,  0.0000, -0.7432, -1.4123, -0.5496, -0.0000,  0.0000, -0.4169,
    #         -1.7772, -0.9935, -0.3791, -0.0000,  0.0000,  0.0000, -1.1870, -0.6619,
    #         -0.2097, -0.0000, -0.0000,  0.0000]
    
    
    c_1: float = 1.3753
    c_2: float = 1.2447
    c_3: float = 1.0474
    c_4: float = 0.5495
    c_5: float = 0.4873
    c_6: float = 0.7254
    c_7: float = 0
    c_8: float = -0.4877
    c_9: float = 0
    c_10: float = 0
    c_11: float = -0.7432
    c_12: float = -1.4123
    c_13: float = -0.5496
    c_14: float = 0
    c_15: float = 0
    c_16: float = -0.4169
    c_17: float = -1.7772
    c_18: float = -0.9935
    c_19: float = -0.3791
    c_20: float = 0
    c_21: float = 0
    c_22: float = 0
    c_23: float = -1.1870
    c_24: float = -0.6619
    c_25: float = -0.2097
    c_26: float = 0
    c_27: float = 0
    c_28: float = 0
    
    
    #######
    #first try
    #        [ 2.3566,  4.8932, -2.4195,  0.0000,  2.7663, -1.7639, -1.3326, -2.0735,
    #          2.4340,  2.9003,  2.4481, -2.1664, -2.3395,  1.7161,  5.5047,  5.9845,
    #          2.2588, -1.4272, -1.7946,  0.2765,  3.1754,  3.0999,  3.8608,  3.9005,
    #          1.7216, -0.5702, -0.0000]
    
    #second try
    #        [ 1.8057,  3.1958,  0.0000,  0.0000,  2.4179,  0.0000,  0.0000,  0.0000,
    #          0.4797,  1.3243,  0.0000, -0.0000,  0.0000,  0.0000,  5.8689,  0.3974,
    #         -0.0000,  0.0000,  0.0000,  0.0000,  4.5259,  2.9360,  2.4411,  0.0000,
    #          0.0000, -0.0000, -0.0000]
    
    
    d_1: float = 1
    d_2: float = 1.8057
    d_3: float = 3.1958
    d_4: float = 0
    d_5: float = 0
    d_6: float = 2.4179
    d_7: float = 0
    d_8: float = 0
    d_9: float = 0
    d_10: float = 0.4797
    d_11: float = 1.3243
    d_12: float = 0
    d_13: float = 0
    d_14: float = 0
    d_15: float = 0
    d_16: float = 5.8689
    d_17: float = 0.3974
    d_18: float = 0
    d_19: float = 0
    d_20: float = 0
    d_21: float = 0
    d_22: float = 4.5259
    d_23: float = 2.9360
    d_24: float = 2.4411
    d_25: float = 0
    d_26: float = 0
    d_27: float = 0
    d_28: float = 0
    
    
    
    def f_2(self,state,t):
        #print(initial_condition)
        x,y = state  # Unpack the state vector
        #return self.b - self.a * x + self.c * x * x
        #return  self.a_1 + (self.a_2 * x * x)/(self.a_3 + x * x ) - (x)/(1+ x + y), (self.b_1)/(1+self.b_2 * x*x*x*x*x) - (y)/(1+x+y)
        #return  (- 1 * x)/(1 + x ), (-y)/(1 + x ) works
        #return  (- 1 * x)/(1 + x), (-y)/(1 + x + x * x)works
        #return  (- 1 * x)/(1 +  x +  x * x), (-y)/(1 + y + x * x ) works
        #return  (- 1 * x)/(1 +  x +  x * x), (-y - x)/(1 + y + x * x ) works
        #return  (- 1 * x)/(1 +  x + x * y +  x * x), (-y - x)/(1 + y + x * x ) works
        #return  (- 1 * x)/(1 +  x + x * y +  x * z), (-y - x)/(1 + y + x * x), (-z - x)/(1 + z + y + z * z )works
        #return  (- 1 * x)/(1 +  x + x * y ), (-y - x)/(1 + y + x * x * x )works
        #return  (- 1 * x)/(1 +  x + x * y ), (-y - x)/(1 + y + x * x * x * x) works
        #return  (- 1 * x)/(1 +  x + x * y ), (-y -x)/(1 + y + x * x * x * x * x) works
        #return   (- 1 * x * x)/(1 +  x + x * y + x * x * x), (-y - x)/(1 + y + x * x * x * x * x) works
        #return self.a_1 + (self.a_2 * x * x)/(self.a_3 + x * x) - (x)/(1+ x + y), (self.b_1)/(1+self.b_2 * x*x*x*x*x) - (y)/(1+x+y)
        #return (-3*x*x + 4*x + 2)/(1 + x + x*x + 0.5*x*x*x + x*x*x*x) did not try yet
        #return [(self.a_1 +self.a_2*x + self.a_3*y+self.a_4*x*x + self.a_5*x*y+
        #         self.a_6*y*y+self.a_7*x*x*x+self.a_8*x*x*y+self.a_9*x*y*y+self.a_10*y*y*y
        #         + self.a_11*x*x*x*x+ self.a_12*x*x*x*y+ self.a_13*x*x*y*y+ self.a_14*x*y*y*y+ self.a_15*y*y*y*y
        #         + self.a_16*x*x*x*x*x+ self.a_17*x*x*x*x*y+ self.a_18*x*x*x*y*y+ self.a_19*x*x*y*y*y+
        #         self.a_20*x*y*y*y*y+ self.a_21*y*y*y*y*y)
        #        / (self.b_1 +self.b_2*x + self.b_3*y+self.b_4*x*x + self.b_5*x*y+
        #         self.b_6*y*y+self.b_7*x*x*x+self.b_8*x*x*y+self.b_9*x*y*y+self.b_10*y*y*y
        #         + self.b_11*x*x*x*x+ self.b_12*x*x*x*y+ self.b_13*x*x*y*y+ self.b_14*x*y*y*y+ self.b_15*y*y*y*y
        #         + self.b_16*x*x*x*x*x+ self.b_17*x*x*x*x*y+ self.b_18*x*x*x*y*y+ self.b_19*x*x*y*y*y+
        #         self.b_20*x*y*y*y*y+ self.b_21*y*y*y*y*y),
        #        (self.c_1 +self.c_2*x + self.c_3*y+self.c_4*x*x + self.c_5*x*y+
        #         self.c_6*y*y+self.c_7*x*x*x+self.c_8*x*x*y+self.c_9*x*y*y+self.c_10*y*y*y
        #         + self.c_11*x*x*x*x+ self.c_12*x*x*x*y+ self.c_13*x*x*y*y+ self.c_14*x*y*y*y+ self.c_15*y*y*y*y
        #         + self.c_16*x*x*x*x*x+ self.c_17*x*x*x*x*y+ self.c_18*x*x*x*y*y+ self.c_19*x*x*y*y*y+
        #         self.c_20*x*y*y*y*y+ self.c_21*y*y*y*y*y)/
        #        (self.d_1 +self.d_2*x + self.d_3*y+self.d_4*x*x + self.d_5*x*y+
        #         self.d_6*y*y+self.d_7*x*x*x+self.d_8*x*x*y+self.d_9*x*y*y+self.d_10*y*y*y
        #         + self.d_11*x*x*x*x+ self.d_12*x*x*x*y+ self.d_13*x*x*y*y+ self.d_14*x*y*y*y+ self.d_15*y*y*y*y
        #         + self.d_16*x*x*x*x*x+ self.d_17*x*x*x*x*y+ self.d_18*x*x*x*y*y+ self.d_19*x*x*y*y*y+
        #         self.d_20*x*y*y*y*y+ self.d_21*y*y*y*y*y)
        #       ]
        return [(self.a_1 +self.a_2*x + self.a_3*y+self.a_4*x*x + self.a_5*x*y+
                 self.a_6*y*y+self.a_7*x*x*x+self.a_8*x*x*y+self.a_9*x*y*y+self.a_10*y*y*y
                 + self.a_11*x*x*x*x+ self.a_12*x*x*x*y+ self.a_13*x*x*y*y+ self.a_14*x*y*y*y+ self.a_15*y*y*y*y
                 + self.a_16*x*x*x*x*x+ self.a_17*x*x*x*x*y+ self.a_18*x*x*x*y*y+ self.a_19*x*x*y*y*y+
                 self.a_20*x*y*y*y*y+ self.a_21*y*y*y*y*y+ self.a_22 *x*x*x*x*x*x +
                 self.a_23*x*x*x*x*x*y + self.a_24*x*x*x*x*y*y + self.a_25*x*x*x*y*y*y
                 + self.a_26*x*x*y*y*y*y + self.a_27*x*y*y*y*y*y + self.a_28*y*y*y*y*y*y  )
                / (self.b_1 +self.b_2*x + self.b_3*y+self.b_4*x*x + self.b_5*x*y+
                 self.b_6*y*y+self.b_7*x*x*x+self.b_8*x*x*y+self.b_9*x*y*y+self.b_10*y*y*y
                 + self.b_11*x*x*x*x+ self.b_12*x*x*x*y+ self.b_13*x*x*y*y+ self.b_14*x*y*y*y+ self.b_15*y*y*y*y
                 + self.b_16*x*x*x*x*x+ self.b_17*x*x*x*x*y+ self.b_18*x*x*x*y*y+ self.b_19*x*x*y*y*y+
                 self.b_20*x*y*y*y*y+ self.b_21*y*y*y*y*y+ self.b_22*x*x*x*x*x*x +
                 self.b_23*x*x*x*x*x*y + self.b_24*x*x*x*x*y*y + self.b_25*x*x*x*y*y*y
                 + self.b_26*x*x*y*y*y*y + self.b_27*x*y*y*y*y*y + self.b_28*y*y*y*y*y*y),
                (self.c_1 +self.c_2*x + self.c_3*y+self.c_4*x*x + self.c_5*x*y+
                 self.c_6*y*y+self.c_7*x*x*x+self.c_8*x*x*y+self.c_9*x*y*y+self.c_10*y*y*y
                 + self.c_11*x*x*x*x+ self.c_12*x*x*x*y+ self.c_13*x*x*y*y+ self.c_14*x*y*y*y+ self.c_15*y*y*y*y
                 + self.c_16*x*x*x*x*x+ self.c_17*x*x*x*x*y+ self.c_18*x*x*x*y*y+ self.c_19*x*x*y*y*y+
                 self.c_20*x*y*y*y*y+ self.c_21*y*y*y*y*y + self.c_22*x*x*x*x*x*x +
                 self.c_23*x*x*x*x*x*y + self.c_24*x*x*x*x*y*y + self.c_25*x*x*x*y*y*y
                 + self.c_26*x*x*y*y*y*y + self.c_27*x*y*y*y*y*y + self.c_28*x*y*y*y*y*y)/
                (self.d_1 +self.d_2*x + self.d_3*y+self.d_4*x*x + self.d_5*x*y+
                 self.d_6*y*y+self.d_7*x*x*x+self.d_8*x*x*y+self.d_9*x*y*y+self.d_10*y*y*y
                 + self.d_11*x*x*x*x+ self.d_12*x*x*x*y+ self.d_13*x*x*y*y+ self.d_14*x*y*y*y+ self.d_15*y*y*y*y
                 + self.d_16*x*x*x*x*x+ self.d_17*x*x*x*x*y+ self.d_18*x*x*x*y*y+ self.d_19*x*x*y*y*y+
                 self.d_20*x*y*y*y*y+ self.d_21*y*y*y*y*y+ self.d_22*x*x*x*x*x*x +
                 self.d_23*x*x*x*x*x*y + self.d_24*x*x*x*x*y*y + self.d_25*x*x*x*y*y*y
                 + self.d_26*x*x*y*y*y*y + self.d_27*x*y*y*y*y*y + self.d_28*x*y*y*y*y*y)
               ]
        
        
        
        
    def solution(self):
        
        num_features = len(self.list_initial_conditions[0])
        u_original_main = []
        initial_cond_main = []
        t_scaled_main = []
        
        for i in range(0, len(self.list_initial_conditions)):
            #print(i)
            initial_condition = self.list_initial_conditions[i]
            
            u_original = odeint(self.f_2, initial_condition, self.t)
            
            
            u_original_main.append(torch.from_numpy(self.scaling_factor*u_original))
            
            vector_initial_cond = torch.mul(torch.tensor(initial_condition)
                                            ,torch.ones_like(torch.from_numpy(u_original)))
            vector_initial_cond = torch.div(vector_initial_cond,
                                            max(initial_condition, key=abs))
            
            initial_cond_main.append(vector_initial_cond)
            
            
            
            t_scaled_main.append(time_scaling_func(self.t))
            
        t_scaled_main = torch.stack(t_scaled_main, dim=0)
        
        u_original_main = torch.stack(u_original_main, dim=0)
        
        initial_cond_main = torch.stack(initial_cond_main, dim=0)
        
        t_scaled_main = torch.reshape(t_scaled_main,
                                      (len(self.list_initial_conditions)*self.t.size()[0],1))
        
        u_original_main = torch.reshape(u_original_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))
        
        initial_cond_main = torch.reshape(initial_cond_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))

        return t_scaled_main, initial_cond_main, u_original_main, u_original
            


###################
###################
###################




def Fitz_Hugh_2(state,t):
    v, w = state
    a_1:float = -0.33333
    a_2:float = 0.04
    a_3:float = -0.028
    b_1:float = 0.5
    b_2:float = 0.032
    #print(initial_condition)
      # Unpack the state vector
    return v - w + a_1 * v*v*v + b_1 , a_2 * v + a_3 * w + b_2





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


@dataclass
class rational_nD:
    t: torch.tensor
    list_initial_conditions: list
    scaling_factor: float
    #function_name: str = 'two_D_oscillator'
    
    
    ## for the case of x^5, t = torch.linspace(0, 30, 150)
    a_1: float = 0.2
    a_2: float = 0.2
    a_3: float = 0.2
    b_1: float = 0.8
    b_2: float = 3
    
    #for the case of x^6 paper
    a_1: float = 0.1
    a_2: float = 0.1
    a_3: float = 0.1
    b_1: float = 0.8
    b_2: float = 3
    
    
    
    ####
    
    
    
    #vs = 0.76 (μMh−1) k s = 0.38 (h−1)
    #vm = 0.65 (μMh−1) k1 = 1.9 (h−1)
    #vd = 0.95 (μMh−1) k2 = 1.3 (h−1)
    #V1 = 3.2 (μMh−1) K d = 0.2 (μM)
    #V2 = 1.58 (μMh−1) K I = 1 (μM)
    #V3 = 5 (μMh−1) K m = 0.5 (μM)
    #V4 = 2.5 (μMh−1) K1,2,3,4 = 2 (μM)
    
    
    #################
    #k_i: float = 1
    #k_d: float = 0.2
    #k_2: float = 1.3
    #k_1: float = 1.9
    #k_s: float = 0.38
    #k_m: float = 0.5
    #V1: float = 3.2
    #V2: float = 1.58
    #V3: float = 5
    #V4: float = 2.5
    #vm: float = 0.65
    #vd: float = 0.95
    #vs: float = 0.76
    
    def f_2(self,x,t):
        #print(initial_condition)
        #x,y = state  # Unpack the state vector
        # the solutin of this equation is done above
        
        #dx = - x[0] - x[0]/(1+x[0]) 
        dx = -x[0] + 0.5 
        
        return np.array([dx])
        
        
        #return   [self.a_1 + (self.a_2 * x * x)/(self.a_3 + x * x )
        #          - (x)/(1+ x + y),
        #          (self.a_1 + self.b_1)/(1+self.b_2 * x*x) - (y)/(1+x+y)
        #          ]
        
    def f_3(self,state,t):
        
        
        x,y = state
        
        
        return  (-x+0.5) + (-x)/(1 + y*y), (-x+0.5) + (-y)/(1 +  x*x)
    
    
    
    def two_D_oscillator(self,state,t):
        #print(initial_condition)
        x, y = state  # Unpack the state vector
        a_1:float = -0.1
        a_2:float = 2
        a_3:float = -2
        a_4:float = -0.1
        b_1:float = 0
        b_2:float = 0
        return -0.1 * x + 2 * y + 0, -2 * x + 0.1 * y + 0
    
    
    def three_D_nonlinear(self,state,t):
        #print(initial_condition)
        x,y,z = state  # Unpack the state vector
        
        a_11:float = 1
        a_21:float = 1
        a_31:float = -3
        a_32:float = -1
        a_33:float = -2.67
        a_34:float = 0
        a_35:float = -1
        
        
        return a_11*y,  a_21*z, a_31*x + a_32*y + a_33*z + a_35*x*z 
    
    def lorenz(self,state,t):
        #lorenz system
        #print(initial_condition)
        x, y, z = state  # Unpack the state vector
                
        rho = 28.0
        sigma = 10.0
        beta = 8.0 / 3.0
        
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
    
    
    def Fitz_Hugh_Nagumo(self,state,t):
        v, w = state
        a_1:float = -0.33333
        a_2:float = 0.04
        a_3:float = -0.028
        b_1:float = 0.5
        b_2:float = 0.032
        #print(initial_condition)
          # Unpack the state vector
        return v - w + a_1 * v*v*v + b_1 , a_2 * v + a_3 * w + b_2
    
    
    
    
    
    
    def simplification(self):
        
        x, y = symbols('x y')
        
        
        return [print(cancel(self.a_1 + (self.a_2 * x * x)/(self.a_3 + x * x ) - (x)/(1+ x + y))),
                print(cancel((self.b_1)/(1+self.b_2 * x*x*x*x) - (y)/(1+x+y)))
                ]
    

    
        
        
    def solution(self):
        
        num_features = len(self.list_initial_conditions[0])
        u_original_main = []
        initial_cond_main = []
        t_scaled_main = []
        
        for i in range(0, len(self.list_initial_conditions)):
            #print(i)
            initial_condition = self.list_initial_conditions[i]
            
            
            #Fitz_Hugh_2
            #self.lorenz
            u_original = odeint(self.two_D_oscillator, initial_condition, self.t)
            

            
            u_original_main.append(torch.from_numpy(self.scaling_factor*u_original))
            
            
            
            vector_initial_cond = torch.mul(torch.tensor(initial_condition)
                                            ,torch.ones_like(torch.from_numpy(u_original)))
            vector_initial_cond = torch.div(vector_initial_cond,
                                            max(initial_condition, key=abs))
            
            initial_cond_main.append(vector_initial_cond)
            
            
            
            t_scaled_main.append(time_scaling_func(self.t))
        
            
        t_scaled_main = torch.stack(t_scaled_main, dim=0)
        
       
        t_scaled_main_rk4 = t_scaled_main
        
        u_original_rk4 = u_original_main
        
        u_original_main = torch.stack(u_original_main, dim=0)
        
        
        
        initial_cond_main = torch.stack(initial_cond_main, dim=0)
        
        initial_cond_main_rk4 = initial_cond_main
        
        
        
        t_scaled_main = torch.reshape(t_scaled_main,
                                      (len(self.list_initial_conditions)*self.t.size()[0],1))
        
        u_original_main = torch.reshape(u_original_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))
        
        initial_cond_main = torch.reshape(initial_cond_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))
        
        
        
        return (t_scaled_main, initial_cond_main, u_original_main,
                            u_original, u_original_rk4, t_scaled_main_rk4, initial_cond_main_rk4)
            
###################
###################
###################



@dataclass
class rational_7D:
    t: torch.tensor
    list_initial_conditions: list
    scaling_factor: float = 1
    
    ####
    
    c_1: float = 2.5
    c_2: float = -100
    c_3: float = 13.6769
    d_1: float = 200
    d_2: float = 13.6769
    d_3: float = -6
    d_4: float = -6
    e_1: float = 6
    e_2: float = -64
    e_3: float = 6
    e_4: float = 16
    l_1: float = 64
    l_2: float = -13
    l_3: float = 13
    l_4: float = -16
    l_5: float = -100
    g_1: float = 1.3
    g_2: float = -3.1
    h_1: float = -200
    h_2: float = 13.6769
    h_3: float = 128
    h_4: float = -1.28
    h_5: float = -32
    j_1: float = 6
    j_2: float = -18
    j_3: float = -100
    
    
    #vs = 0.76 (μMh−1) k s = 0.38 (h−1)
    #vm = 0.65 (μMh−1) k1 = 1.9 (h−1)
    #vd = 0.95 (μMh−1) k2 = 1.3 (h−1)
    #V1 = 3.2 (μMh−1) K d = 0.2 (μM)
    #V2 = 1.58 (μMh−1) K I = 1 (μM)
    #V3 = 5 (μMh−1) K m = 0.5 (μM)
    #V4 = 2.5 (μMh−1) K1,2,3,4 = 2 (μM)
    
    
    #################
    #k_i: float = 1
    #k_d: float = 0.2
    #k_2: float = 1.3
    #k_1: float = 1.9
    #k_s: float = 0.38
    #k_m: float = 0.5
    #V1: float = 3.2
    #V2: float = 1.58
    #V3: float = 5
    #V4: float = 2.5
    #vm: float = 0.65
    #vd: float = 0.95
    #vs: float = 0.76
    
    
    ##################
    ################## from the original reference
    
    J_0: float = 2.5
    
    k_1: float = 100
    k_2: float = 6
    k_3: float = 16
    k_4: float = 100
    k_5: float = 1.28
    k_6: float = 12
    
    k: float = 1.8
    kappa: float = 13
    q_s: float = 4
    k_b: float = 0.52
    phi: float = 0.1
    N: float = 1
    A: float = 4
    
    
    
    def GlycolyticOsci_model(self,x,t):
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
    
    
    
    
    
    
    
    
    
    def f_2(self,state,t):
        #print(initial_condition)
        x_1,x_2,x_3,x_4,x_5,x_6,x_7 = state  # Unpack the state vector
        # the solutin of this equation is done above
        dx_1 = self.J_0 - (self.k_1 * x_1 * x_6)/(1 +(x_6/self.k_b)**self.q_s)
        dx_2 = (2*self.k_1*x_1*x_6)/(1 + (x_6/self.k_b)**self.q_s ) - self.k_2*x_2*(self.N-x_5) - self.k_6*x_2*x_5
        dx_3 = self.k_2*x_2*(self.N-x_5) - self.k_3*x_3*(self.A - x_6)
        dx_4 = self.k_3*x_3*(self.A - x_6) - self.k_4*x_4*x_5 - self.kappa*(x_4-x_7)
        dx_5 = self.k_2*x_2*(self.N - x_5) - self.k_4*x_4*x_5 - self.k_6*x_2*x_5
        dx_6 = -(2*self.k_1*x_1*x_6)/(1+ (x_6/self.k_1)**self.q_s ) + 2*self.k_3*x_3*(self.A - x_6) - self.k_5*x_6,
        dx_7 = self.phi*self.kappa*(x_4-x_7) - self.k*x_7

        return  np.array([dx_1,dx_2,dx_3,dx_4,dx_5,dx_6,dx_7])
                
                #[self.J_0 - (self.k_1 * x_1 * x_6)/(1 +(x_6/self.k_b)**self.q_s),
                 
                 #(2*self.k_1*x_1*x_6)/(1 + (x_6/self.k_b)**self.q_s ) - self.k_2*x_2*(self.N-x_5) -self.k_6*x_2*x_5,
                 
                 #self.k_2*x_2*(self.N-x_5) - self.k_3*x_3*(self.A - x_6),
                 
                 #self.k_3*x_3*(self.A-x_6) - self.k_4*x_4*x_5 - self.kappa*(x_4-x_7),
                 
                 #self.k_2*x_2*(self.N-x_5) - self.k_4*x_4*x_5 - self.k_6*x_2*x_5,
                 
                 #-(2*self.k_1*x_1*x_6)/(1+ (x_6/self.k_1)**self.q_s ) + 2*self.k_3*x_3*(self.A- x_6) - self.k_5*x_6,
                 
                 #self.phi*self.kappa*(x_4-x_7) - self.k*x_7]
    
                
    
    #[self.c_1 + (self.c_2 * x * q)/(1 +self.c_3* q*q*q*q*q*q )
    #,(self.d_1*x*q)/(1+self.d_2*q*q*q*q) + self.d_3*y- self.d_4*y*v
    #,self.e_1*y+self.e_2*z+self.e_3*y*v+ self.e_4*z*q
    #,self.l_1*z+ self.l_2*w+ self.l_3*p+ self.l_4*z*q+ self.l_5*w*v
    #,self.g_1*w + self.g_2*p
    #,(self.h_1*x*q)/(1+self.h_2*q*q*q*q)+ self.h_3*z+ self.h_5*q+ self.h_4*z*v
    #,self.j_1*y+self.j_2*y*v+self.j_3*w*v
    #]
                  
     
    
    
    def simplification(self):
        
        x,y,z,w,p,q,v = symbols('x y z w p q v')
        
        
        
        #print(cancel(self.a_1 + (self.a_2 * x * x)/(self.a_3 + x * x ) - (x)/(1+ x + y)))
        
        #print(cancel((self.b_1)/(1+self.b_2 * x*x*x*x*x) - (y)/(1+x+y)))
        
        
        return [print(cancel( self.c_1 + (self.c_2 * x * q)/(1 +self.c_3* q*q*q*q*q*q ))),
                print(cancel(( self.d_1*x*q)/(1+self.d_2*q*q*q*q) + self.d_3*y- self.d_4*y*v)),
                print(cancel( self.e_1*y+self.e_2*z+self.e_3*y*v+ self.e_4*z*q )),
                print(cancel( self.l_1*z + self.l_2*w + self.l_3*p + self.l_4*z*q + self.l_5*w*v )),
                print(cancel( self.g_1*w + self.g_2*p )),
                print(cancel( (self.h_1*x*q)/(1+self.h_2*q*q*q*q)+ self.h_3*z+ self.h_5*q+ self.h_4*z*v )),
                print(cancel( self.j_1*y+self.j_2*y*v+self.j_3*w*v ))
                ]
    

    
        
        
    def solution(self):
        
        num_features = len(self.list_initial_conditions[0])
        u_original_main = []
        initial_cond_main = []
        t_scaled_main = []
        
        for i in range(0, len(self.list_initial_conditions)):
            #print(i)
            initial_condition = self.list_initial_conditions[i]
            
            #print(initial_condition)
            
            #self.f_2
            
            u_original = odeint(self.GlycolyticOsci_model, initial_condition, self.t)
            
            #u_original_2 = odeint(self.f_2, initial_condition, self.t)

            
            
            u_original_main.append(torch.from_numpy(self.scaling_factor*u_original))
            
            vector_initial_cond = torch.mul(torch.tensor(initial_condition)
                                            ,torch.ones_like(torch.from_numpy(u_original)))
            vector_initial_cond = torch.div(vector_initial_cond,
                                            max(initial_condition, key=abs))
            
            initial_cond_main.append(vector_initial_cond)
            
            
            
            t_scaled_main.append(time_scaling_func(self.t))
            
        t_scaled_main = torch.stack(t_scaled_main, dim=0)
        
        u_original_main = torch.stack(u_original_main, dim=0)
        
        initial_cond_main = torch.stack(initial_cond_main, dim=0)
        
        t_scaled_main = torch.reshape(t_scaled_main,
                                      (len(self.list_initial_conditions)*self.t.size()[0],1))
        
        u_original_main = torch.reshape(u_original_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))
        
        initial_cond_main = torch.reshape(initial_cond_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))

        return t_scaled_main, initial_cond_main, u_original_main, u_original



###################
###################
###################


@dataclass
class rational:
    a: float
    b: float
    t: torch.tensor
    list_initial_conditions: list
    scaling_factor: float = 1
    c: float = 1.5
    d: float = 0.7
    
    def f_2(self,state,t):
        #print(initial_condition)
        x = state  # Unpack the state vector
        #return self.b - self.a * x + self.c * x * x
        return  (-0.450 - 0.9 *x)/(0.6 + 0.5 * x )
        #return -0.459 - 0.9 * x
        
    @staticmethod
    def f_3(self,state,t):
        
        #print(initial_condition)
        x, y, z = state  # Unpack the state vector
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z
    
    
    def solution(self):
        
        def f(state, t):
            x, y, z = state  # Unpack the state vector
            return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z
        
        num_features = len(self.list_initial_conditions[0])
        u_original_main = []
        initial_cond_main = []
        t_scaled_main = []
        
        for i in range(0, len(self.list_initial_conditions)):
            #print(i)
            initial_condition = self.list_initial_conditions[i]
            
            u_original = odeint(self.f_2, initial_condition, self.t)
            
            
            u_original_main.append(torch.from_numpy(self.scaling_factor*u_original))
            
            vector_initial_cond = torch.mul(torch.tensor(initial_condition)
                                            ,torch.ones_like(torch.from_numpy(u_original)))
            vector_initial_cond = torch.div(vector_initial_cond,
                                            4)
            
            initial_cond_main.append(vector_initial_cond)
            
            
            
            t_scaled_main.append(time_scaling_func(self.t))
            
        t_scaled_main = torch.stack(t_scaled_main, dim=0)
        
        u_original_main = torch.stack(u_original_main, dim=0)
        
        initial_cond_main = torch.stack(initial_cond_main, dim=0)
        
        t_scaled_main = torch.reshape(t_scaled_main,
                                      (len(self.list_initial_conditions)*self.t.size()[0],1))
        
        u_original_main = torch.reshape(u_original_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))
        
        initial_cond_main = torch.reshape(initial_cond_main,
                                        (len(self.list_initial_conditions)*self.t.size()[0],num_features))

        return t_scaled_main, initial_cond_main, u_original_main, u_original
            












#####################
#####################
#####################












def exp_function(initial_condition: float, t: torch.tensor)-> torch.tensor:
    
    initial_condition = torch.tensor(initial_condition)

    u = 0.6 * torch.exp(0.2*t) + 0.4 * torch.exp(-0.5*t)
    
    coords = t.reshape(-1,1)
    
    return coords, u.view(-1,1)
