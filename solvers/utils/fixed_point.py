import torch
import random

class StandardFixedPointSolver:
    def __init__(self, max_iter=50, tol=1e-3, solver="anderson"):
        self.tol = tol
        self.max_iter = max_iter
        self.solver=solver
        
    def forward(self, f, x):
        if self.solver=="anderson":
            fixed_point = self.anderson(f, x)
        elif self.solver=="naive_forward":
            fixed_point = self.naive_forward(f, x)
        return fixed_point
    
    def naive_forward(self, f, x):
        for k in range(0, self.max_iter):
            previous = x
            x = f(x)
            
            diff = ((x - previous).norm().item() / (1e-5 + x.norm().item()))
            if diff < self.tol:
                break 

        return x
        
    def anderson(self, f, x0, m=5, lam=1e-4, beta = 1.0):
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
        
        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1
        
        res = []
        for k in range(2, self.max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
            
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
            res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
            if (res[-1] < self.tol):
                break  

        return X[:,k%m].view_as(x0)

class FixedPointSolver:
    def __init__(self, max_iter=50, tol=1e-3, solver="anderson"):
        self.tol = tol
        self.max_iter = max_iter
        self.solver=solver
        
    def forward(self, f, x, u):
        if self.solver=="anderson":
            fixed_point = self.anderson(f, x)
        elif self.solver=="naive_forward":
            fixed_point = self.naive_forward(f, x, u)
        return fixed_point
    
    def naive_forward(self, f, x, u):
        for k in range(0, self.max_iter):
            previous = x
            x,u = f(x,u)
            
            diff = ((x - previous).norm().item() / (1e-5 + x.norm().item()))
            if diff < self.tol:
                break 

        return x, u
        
    def anderson(self, f, x0, m=5, lam=1e-4, beta = 1.0):
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
        
        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1
        
        res = []
        for k in range(2, self.max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
            
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
            res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
            if (res[-1] < self.tol):
                break  

        return X[:,k%m].view_as(x0), res
        
class RandomBufferFixedPointSolver:
    def __init__(self, max_iter=50, tol=1e-3, solver="anderson", buffer_size=1):
        self.tol = tol
        self.max_iter = max_iter
        self.solver=solver
        self.buffer_size = buffer_size
        
    def forward(self, f, x, u):
        if self.solver=="anderson":
            fixed_point = self.anderson(f, x)
        elif self.solver=="naive_forward":
            fixed_point = self.naive_forward(f, x, u)
        return fixed_point
        
    def update_buffer(self, buffer_step, buffer_random_sample, buffer_iteration, x, u, iteration):
        random_sample = random.uniform(0, 1)
        
        if iteration < self.buffer_size-1:
            buffer_step.append([x,u])
            buffer_random_sample.append(random_sample)
            buffer_iteration.append(iteration)
            
            buffer_step, buffer_random_sample, buffer_iteration = (list(t) for t in zip(*sorted(zip(buffer_step, buffer_random_sample, buffer_iteration), key=lambda pair: pair[1])))
        elif buffer_random_sample[0] < random_sample:
            buffer_step[0] = [x,u]
            buffer_random_sample[0] = random_sample
            buffer_iteration[0] = iteration
            
            buffer_step, buffer_random_sample, buffer_iteration = (list(t) for t in zip(*sorted(zip(buffer_step, buffer_random_sample, buffer_iteration), key=lambda pair: pair[1])))
        
        return buffer_step, buffer_random_sample, buffer_iteration
        
    
    def naive_forward(self, f, x, u):
        buffer_step = [[x,u]] 
        buffer_random_sample = [random.uniform(0, 1)]
        buffer_iteration = [0]
        for k in range(0, self.max_iter):
            previous = x
            x,u = f(x,u)
            
            buffer_step, buffer_random_sample, buffer_iteration = self.update_buffer(buffer_step, buffer_random_sample, buffer_iteration, x, u, k)

            diff = ((x - previous).norm().item() / (1e-5 + x.norm().item()))
            if diff < self.tol:
                break  
                
        buffer_step, buffer_random_sample, buffer_iteration = (list(t) for t in zip(*sorted(zip(buffer_step, buffer_random_sample, buffer_iteration), key=lambda pair: pair[2])))        
        return x, u, buffer_step
        
    def anderson(self, f, x0, m=5, lam=1e-4, beta = 1.0):
        buffer_step = [x0] 
        buffer_random_sample = [random.uniform(0, 1)]
        buffer_iteration = [0]
    
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
        
        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1
        
        res = []
        for k in range(2, self.max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
            
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
            
            buffer_step, buffer_random_sample, buffer_iteration = self.update_buffer(buffer_step, buffer_random_sample, buffer_iteration, X[:,k%m].view_as(x0), k)
            
            res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
            if (res[-1] < self.tol):
                break  

        buffer_step, buffer_random_sample, buffer_iteration = (list(t) for t in zip(*sorted(zip(buffer_step, buffer_random_sample, buffer_iteration), key=lambda pair: pair[2])))        
        return X[:,k%m].view_as(x0), buffer_step