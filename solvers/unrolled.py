import os
import torch
from networks.u_net import UNetRes as net
from networks.utils import utils_model

class Unrolled(torch.nn.Module):
    def __init__(self, linear_operator, nb_channels=3, nb_block=6, recurrent=True, pretrained=True, penalty_initial_val=0.1, sigma_initial_val=0.1, device='cpu'):
        super(Unrolled,self).__init__()
        self.nb_block=nb_block
        self.linear_op = linear_operator
        self.recurrent = recurrent
        self.pretrained = pretrained
        self.device = device
        
        if self.recurrent:
            if self.pretrained:
                self.nonlinear_op = net(in_nc=nb_channels+1, out_nc=nb_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
                self.nonlinear_op.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__))+"/../ckpts/pretrained_denoiser.pth", map_location=torch.device('cpu')), strict=True)
            else:
                self.nonlinear_op = net(in_nc=nb_channels, out_nc=nb_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
        else:
            self.nonlinear_op = torch.nn.ModuleList()
            for i in range(nb_block):
                if self.pretrained:
                    self.nonlinear_op += [net(in_nc=nb_channels+1, out_nc=nb_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")]
                    self.nonlinear_op[-1].load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__))+"/../../ckpts/pretrained_denoiser.pth", map_location=torch.device('cpu')), strict=True)
                else:
                    self.nonlinear_op += [net(in_nc=nb_channels, out_nc=nb_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")]
                    
        self.penaltys = torch.nn.ParameterList()
        self.sigmas = torch.nn.ParameterList()
        for i in range(nb_block):
            self.penaltys += [torch.nn.Parameter(data=torch.tensor(penalty_initial_val), requires_grad=True)]
            if pretrained:
                self.sigmas += [torch.nn.Parameter(data=torch.tensor(sigma_initial_val), requires_grad=True)]

    def single_block(self, x, y, u, iteration):
        if self.pretrained:
            img_noise = torch.cat((x+u, self.sigmas[iteration].repeat(x.shape[0], 1, x.shape[2], x.shape[3])), dim=1).to(self.device)
            if self.recurrent:
                denoised = utils_model.test_mode(self.nonlinear_op, img_noise, mode=2, refield=32, min_size=256, modulo=16)
            else:
                denoised = utils_model.test_mode(self.nonlinear_op[iteration], img_noise, mode=2, refield=32, min_size=256, modulo=16)
        else:
            if self.recurrent:
                denoised = utils_model.test_mode(self.nonlinear_op, x+u, mode=2, refield=32, min_size=256, modulo=16)
            else:
                denoised = utils_model.test_mode(self.nonlinear_op[iteration], x+u, mode=2, refield=32, min_size=256, modulo=16)

        data_term = self.linear_op.proximal_solution(denoised-u,y,self.penaltys[iteration], iteration)
        u = u + (data_term - denoised)

        return data_term, u

    def forward(self, y):        
        x = self.linear_op.beforehand(y)
        u = torch.zeros_like(x)
        for b in range(self.nb_block):
            x,u = self.single_block(x, y, u, b)
        return x
