



import torch
from root_classes import Constraint
from typing import List




class LeastSquares(Constraint):
    def __init__(self) -> None:
        """Least Squares Constraint solved by QR decomposition"""
        super().__init__()

    def fit(self, sparse_thetas, time_derivs):
        
                
        Q, R = torch.qr(sparse_thetas)
        
        coeff_vectors = torch.inverse(R) @ Q.T @ time_derivs
        
        
        return coeff_vectors
    
    


