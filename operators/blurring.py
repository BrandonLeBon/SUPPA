import torch
import operators.utils.downsampling_utils as dwn
from operators.utils.fourier_utils import kernel_fft
from operators.utils import utils_sisr as sr
from operators.operator import LinearOperator

class GaussianBlur(LinearOperator):
    def __init__(self, kernel_sigma=5.0, nb_channels=3, pad_mode='circular', device='cpu'):
        super(GaussianBlur, self).__init__()
        self.device = device
        self.pad_mode=pad_mode
        
        self.kernel, self.center = dwn.gen_dwn_kernel2D(1, dwn.gauss3s_filter, filter_param=kernel_sigma, antialias=False, cross_correlation=True)
        self.conv_ = torch.nn.Conv2d(1, 1, kernel_size=self.kernel.shape, stride=1, padding=0)
        self.conv_.weight.requires_grad = False
        self.conv_.bias.requires_grad = False
        self.conv_.weight[0,0] = self.kernel
        self.conv_.bias[:] = 0
        
        pad_left_top = self.center
        pad_right_bottom = self.kernel.shape[0] - self.center - 1
        self.pad = (pad_left_top,pad_right_bottom,pad_left_top,pad_right_bottom)
        self.pad_flip = (pad_right_bottom,pad_left_top,pad_right_bottom,pad_left_top)
        
        self.kernel = self.kernel.to(device=device)

    def forward(self, x):
        x = torch.nn.functional.pad(x, self.pad, mode=self.pad_mode)
        x_shape = x.shape

        x = self.conv_(x.view(x_shape[0]*x_shape[1], 1, x_shape[2], x_shape[3]))

        return x.view(x_shape[0:2] + x.shape[2:])

    def adjoint(self, x):
        x = torch.nn.functional.pad(x, self.pad_flip, mode=self.pad_mode)
        x_shape = x.shape

        k = self.kernel.flip(0, 1).to(self.device)
        k = k.view(1,1,k.shape[0],k.shape[1])
        x = torch.nn.functional.conv2d(x.view(x_shape[0]*x_shape[1], 1, x_shape[2], x_shape[3]), k)
        
        return x.view(x_shape[0:2] + x.shape[2:])
        
    def beforehand(self, x):
        return x
        
    def proximal_solution(self, x, input, penalty, iteration):
        if iteration == 0:
            k = self.kernel.repeat(x.shape[0], 1, 1, 1)

            #compute optical transfer function (Fourier transform of the convolution kernel adapted to the image size)
            self.FB = kernel_fft(k, x.shape[-1], x.shape[-2])

            #precompute other images in Fourier required for solving the proximal operator of blurring data term.
            FBC, F2B, FBFy = sr.pre_calculate_cmplx(input, self.FB, 1, self.device)
            self.FBC = FBC.to(self.device)
            self.F2B = F2B.to(self.device)
            self.FBFy = FBFy.to(self.device)

        # compute solution for proximal operator of data term
        solution = sr.data_solution_cmplx(x, self.FB, self.FBC, self.F2B, self.FBFy, penalty, 1, self.device)
        
        return solution