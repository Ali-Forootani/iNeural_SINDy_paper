#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:08:22 2022

@author: forootani
"""

from abc import ABCMeta, abstractstaticmethod
from scipy.integrate import odeint, ode, solve_ivp
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utiles import normalized_data, time_scaling_func





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
        
    def run(self):
            
        dyn_sys_obj = self.function_data()
        
        t_scaled_main, initial_cond_main, u_original_main, u_original = self.data_prepration(dyn_sys_obj)
        #u = odeint(dyn_sys_obj.Diff_Equation,self.list_initial_conditions[0],self.t)
            
        return t_scaled_main, initial_cond_main, u_original_main, u_original
            
            
            
            
        
        
        
if __name__ == "__main__":
    initial_conditions = [[-8,7,27],[-6,6,25],[-9,8,22]]
    
    t = torch.linspace(0, 10, 400)
    
    y = Factory_Dyn_Sys(t,initial_conditions,0.1,"lorenz")

    t_scaled_main, initial_cond_main, u_original_main, u_original = y.run()
    
    print(u_original_main)
    