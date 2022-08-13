import numpy as np
import torch
from operators.operator import LinearOperator

class Mosaicing(LinearOperator):
    def __init__(self, device='cpu'):
        super(Mosaicing, self).__init__()
        self.kernel = torch.empty(3, 1, 1)
        self.device = device

    def create_bayer_kernel(self, x):
        kernel = torch.zeros_like(x).to(self.device)
        
        kernel[:,0,0::2,0::2] = 1
        kernel[:,1,1::2,0::2] = 1
        kernel[:,1,0::2,1::2] = 1
        kernel[:,2,1::2,1::2] = 1

        return kernel
        
    def forward(self, x):
        if self.kernel.size() != x[0].size():
            self.kernel = self.create_bayer_kernel(x)
        return torch.mul(self.kernel, x)

    def adjoint(self, x):
        return self.forward(x)
        
    def beforehand(self, x):
        return x
        
    def proximal_solution(self, x, input, penalty, iteration):
        if self.kernel.size() != x[0].size():
            self.kernel = self.create_bayer_kernel(x)
            
        if penalty == 0 or penalty == -1:
            penalty += 0.000001 #avoid dividing by zero
        solution = (1 / (self.kernel + penalty)) * (self.adjoint(input) + penalty * x)
        
        return solution
        
