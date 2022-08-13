import os
import torch
import torch.autograd as autograd
from networks.u_net import UNetRes as net
from networks.utils import utils_model
from solvers.utils.fixed_point import FixedPointSolver, StandardFixedPointSolver

class DEQ(torch.nn.Module):
    def __init__(self, linear_operator, nb_channels=3, pretrained=True, solver='naive_forward', max_iter=50, tol=1e-3, penalty_initial_val=0.1, sigma_initial_val=0.1, device='cpu'):
        super().__init__()
        self.fixed_point_solver = FixedPointSolver(max_iter, tol, solver)
        self.jacobian_fixed_point_solver = StandardFixedPointSolver(max_iter, tol, 'anderson')
        self.max_iter = max_iter
        self.tol = tol
        self.pretrained = pretrained
        self.device = device
        self.linear_op = linear_operator
        self.iteration = 0
        
        if self.pretrained:
            self.nonlinear_op = net(in_nc=nb_channels+1, out_nc=nb_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
            self.nonlinear_op.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__))+"/../ckpts/pretrained_denoiser.pth", map_location=torch.device('cpu')), strict=True)
        else:
            self.nonlinear_op = net(in_nc=nb_channels, out_nc=nb_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        
        self.penalty = torch.nn.Parameter(data=torch.tensor(penalty_initial_val), requires_grad=True)
        #self.penalty = penalty_initial_val
        if pretrained: 
            self.sigma = torch.nn.Parameter(data=torch.tensor(sigma_initial_val), requires_grad=True)
        
    def f(self, x, y, u):
        if self.pretrained:
            img_noise = torch.cat((x+u, self.sigma.repeat(x.shape[0], 1, x.shape[2], x.shape[3])), dim=1).to(self.device)
            denoised = utils_model.test_mode(self.nonlinear_op, img_noise, mode=2, refield=32, min_size=256, modulo=16)
        else:
            denoised = utils_model.test_mode(self.nonlinear_op, x+u, mode=2, refield=32, min_size=256, modulo=16)

        data_term = self.linear_op.proximal_solution(denoised-u,y,self.penalty,self.iteration)
        u = u + (data_term - denoised)

        self.iteration += 1

        return data_term, u
        
    def forward(self, y):
        x = self.linear_op.beforehand(y)
        u = torch.zeros_like(x)
        self.iteration = 0
    
        with torch.no_grad():
            z, u = self.fixed_point_solver.forward(lambda z,u : self.f(z, y, u), x, u)
        z,u = self.f(z,y,u)
        
        z0 = z.clone().detach().requires_grad_()
        u0 = u.clone().detach().requires_grad_()
        f0,u0 = self.f(z0,y,u0)
        
        def backward_hook(grad):
            g = self.jacobian_fixed_point_solver.forward(lambda j : autograd.grad(f0, z0, j, retain_graph=True)[0] + grad, grad)
            return g
                
        z.register_hook(backward_hook)

        return z