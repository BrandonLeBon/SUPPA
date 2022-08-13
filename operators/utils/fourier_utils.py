import torch
import torch.fft
import math
import operators.utils.format_utils as fmt


def gen_frequency_grid_centered(res_x, res_y, device=None):
    x_c = math.ceil((res_x + 1) / 2)
    y_c = math.ceil((res_y + 1) / 2)

    x_range = torch.arange(1-x_c,res_x-x_c+1,device=device) / res_x
    y_range = torch.arange(y_c-1,y_c-res_y-1, -1,device=device) / res_y

    w_x, w_y = torch.meshgrid(x_range, y_range)
    return w_x.transpose(0,1).contiguous(), w_y.transpose(0,1).contiguous()


def gen_frequency_grid(res_x, res_y, half_x_dim=False, device=None):
    x_c = math.ceil((res_x + 1) / 2)
    y_c = math.ceil((res_y + 1) / 2)

    x_range = torch.cat((torch.arange(0,res_x-x_c+1,device=device), torch.arange(1-x_c,0,device=device)), 0) / res_x
    y_range = torch.cat((torch.arange(0, y_c-res_y-1, -1,device=device), torch.arange(y_c-1, 0, -1,device=device)), 0) / res_y

    if half_x_dim:
        x_range = x_range[0:x_c]

    w_x, w_y = torch.meshgrid(x_range, y_range)
    return w_x.transpose(0,1).contiguous(), w_y.transpose(0,1).contiguous()


def gen_gaussian2D(res_x,res_y,sigma,y_sigma_ratio=1,theta=0):
    sigma_x = fmt.to_tensor(sigma).view(-1, 1, 1)
    y_sigma_ratio = fmt.to_tensor(y_sigma_ratio).view(-1, 1, 1)
    theta = fmt.to_tensor(theta).view(-1, 1, 1, 1)

    sigma_y = sigma_x * y_sigma_ratio
    #rotation = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    #inv_S = rotation @ torch.diag_embed(1./torch.tensor([sigma_x, sigma_y])**2) @ rotation.transpose(0,1)
    rotation = torch.cat((torch.cat((torch.cos(theta), -torch.sin(theta)), 3), torch.cat((torch.sin(theta), torch.cos(theta)), 3)), 2)
    inv_S = rotation @ torch.diag_embed(1. / torch.cat((sigma_x, sigma_y), 2) ** 2) @ rotation.transpose(2, 3)
    sigma_inv_x2 = inv_S[:, :, 0:1, 0:1]
    sigma_inv_y2 = inv_S[:, :, 1:2, 1:2]
    sigma_inv_xy = inv_S[:, :, 1:2, 0:1] + inv_S[:, :, 0:1, 1:2]

    x, y = gen_frequency_grid_centered(res_x, res_y)
    x *= res_x
    y *= res_y

    return torch.exp(-.5 * (x ** 2 * sigma_inv_x2 + y ** 2 * sigma_inv_y2 + x * y * sigma_inv_xy))


def gen_gaussian2D_fourier(res_x,res_y,sigma,y_sigma_ratio=1,theta=0.0, half_last_dim=True, device=None):
    sigma = fmt.to_tensor(sigma, device).view(-1,1,1)
    y_sigma_ratio = fmt.to_tensor(y_sigma_ratio, device).view(-1, 1, 1)
    theta = fmt.to_tensor(theta, device).view(-1, 1, 1, 1)

    sigma_x_f = 1/(2 * math.pi * sigma)
    sigma_y_f = 1/(2 * math.pi * sigma * y_sigma_ratio)
    #rotation = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    # inv_S_f = rotation @ torch.diag_embed(1./torch.tensor([sigma_x_f, sigma_y_f])**2) @ rotation.transpose(2,3)
    rotation=torch.cat((torch.cat((torch.cos(theta), -torch.sin(theta)), 3), torch.cat((torch.sin(theta), torch.cos(theta)), 3)), 2)
    inv_S_f = rotation @ torch.diag_embed(1. / torch.cat((sigma_x_f, sigma_y_f), 2)**2) @ rotation.transpose(2, 3)
    sigma_inv_x2_f = inv_S_f[:,:,0:1,0:1]
    sigma_inv_y2_f = inv_S_f[:,:,1:2,1:2]
    sigma_inv_xy_f = inv_S_f[:,:,1:2,0:1] + inv_S_f[:,:,0:1,1:2]

    w_x, w_y = gen_frequency_grid(res_x, res_y, half_last_dim, device=device)
    return torch.exp(-.5 * (w_x**2 * sigma_inv_x2_f + w_y**2 * sigma_inv_y2_f + w_x * w_y * sigma_inv_xy_f))


def gen_var_noise_gauss(imgs, sigma_noise_conv, sigma_gauss, y_stretch_ratio=1, theta=0, inf_t1=0.5, inf_t2=1, inf_pow=1, device=None):
    import torch.fft
    from utils.noise_utils import gen_gaussian_noise, infinite_noise_scale

    res_y, res_x = imgs.shape[-2:]
    sigma_noise_conv = fmt.to_tensor(sigma_noise_conv).view(-1, 1, 1, 1)
    epsilon = 1e-12

    imgs = torch.fft.rfftn(imgs, dim=(2, 3))

    gauss_fourier = gen_gaussian2D_fourier(res_x, res_y, sigma_gauss, y_stretch_ratio, theta, device=device)
    gauss_weight = 1. / (gauss_fourier + epsilon)
    gauss_weight *= sigma_noise_conv / gauss_weight[:, :, 0:1, 0:1]

    #Equivalent to creating real valued gaussian noise (sigma=1) in pixel domain and convert to fourier domain (using half spectrum format).
    noise_real = gen_gaussian_noise(imgs.shape, device=device)
    noise_imag = gen_gaussian_noise(imgs.shape, device=device)
    noise_normal_fourier = math.sqrt(res_x*res_y/2) * torch.complex(noise_imag, noise_real)
    noise_normal_fourier[:,:,int(math.ceil((res_y+1)/2)):, 0] = torch.conj(torch.flip(noise_normal_fourier[:,:,1:int(math.floor((res_y+1)/2)), 0], [2]))

    imgs_noisy = ( imgs + noise_normal_fourier * gauss_weight) * infinite_noise_scale(gauss_weight, inf_t1, inf_t2, inf_pow)
    imgs_noisy = torch.fft.irfftn(imgs_noisy, s=(res_y, res_x), dim=(2, 3))

    imgs_noisy_conv = ( imgs/gauss_weight + noise_normal_fourier ) * sigma_noise_conv
    imgs_noisy_conv = torch.fft.irfftn(imgs_noisy_conv, s=(res_y, res_x), dim=(2, 3))

    return imgs_noisy,imgs_noisy_conv


#For test:
#(assumes res_x, res_y are higher or equal to kernel spatial dimensions)
#center_id_x and center_id_y indicate the index of the kernel center (0,0 being the top left corner).
#center_id_x and center_id_y may have negative values or values above the kernel size.
def kernel_fft(kernel_spatial, res_x, res_y, center_id_x=None, center_id_y=None):
    if center_id_x is None:
        center_id_x = int(kernel_spatial.shape[-1] / 2)
    if center_id_y is None:
        center_id_y = int(kernel_spatial.shape[-2] / 2)

    otf = torch.zeros(kernel_spatial.shape[:-2] + (res_y,res_x), dtype=kernel_spatial.dtype, device = kernel_spatial.device)
    otf[..., :kernel_spatial.shape[-2], :kernel_spatial.shape[-1]].copy_(kernel_spatial)
    otf = torch.roll(otf, (-center_id_y, -center_id_x), dims=(-2, -1))
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf

def kernel_rfft(kernel_spatial, res_x, res_y, center_id_x=None, center_id_y=None):
    if center_id_x is None:
        center_id_x = int(kernel_spatial.shape[-1] / 2)
    if center_id_y is None:
        center_id_y = int(kernel_spatial.shape[-2] / 2)

    otf = torch.zeros(kernel_spatial.shape[:-2] + (res_y,res_x), dtype=kernel_spatial.dtype, device = kernel_spatial.device)
    otf[..., :kernel_spatial.shape[-2], :kernel_spatial.shape[-1]].copy_(kernel_spatial)
    otf = torch.roll(otf, (-center_id_y, -center_id_x), dims=(-2, -1))
    otf = torch.fft.rfftn(otf, dim=(-2, -1))
    return otf


def deconvolve_inf_noise(imgs, otf, sigma_noise_conv, inf_t1=0.5, inf_t2=1, inf_pow=1, device=None):
    from utils.noise_utils import infinite_noise_scale
    epsilon=1e-12
    otf = otf / otf[...,0:1,0:1]
    otf_one_side_size = math.ceil((imgs.shape[3]+1)/2)
    imgs_conv = torch.fft.rfftn(imgs, dim=(2, 3)) / (otf[...,:otf_one_side_size] + epsilon)
    imgs_conv *= infinite_noise_scale(torch.abs(sigma_noise_conv / (otf[...,:otf_one_side_size] + epsilon)), inf_t1, inf_t2, inf_pow)
    imgs_conv = torch.fft.irfftn(imgs_conv, (imgs.shape[2],imgs.shape[3]), dim=(2, 3))
    return imgs_conv


def convolve(imgs, otf, device=None):
    otf = otf / otf[...,0:1,0:1]
    imgs_conv = torch.fft.rfftn(imgs, dim=(2, 3)) * otf[...,:math.ceil((imgs.shape[3]+1)/2)]
    imgs_conv = torch.fft.irfftn(imgs_conv, (imgs.shape[2],imgs.shape[3]), dim=(2, 3))
    return imgs_conv

''''''
def deconvolve_inf_noise_old(imgs, otf, sigma_noise_conv, inf_t1=0.5, inf_t2=1, inf_pow=1, device=None):
    from utils.noise_utils import infinite_noise_scale
    epsilon=1e-12
    otf = otf / otf[...,0:1,0:1]
    imgs_conv = torch.fft.fftn(imgs, dim=(2, 3)) / (otf + epsilon)
    imgs_conv *= infinite_noise_scale(torch.abs(sigma_noise_conv / (otf + epsilon)), inf_t1, inf_t2, inf_pow)
    imgs_conv = torch.real(torch.fft.ifftn(imgs_conv, dim=(2, 3)))
    return imgs_conv


def convolve_old(imgs, otf, device=None):
    otf = otf / otf[...,0:1,0:1]
    imgs_conv = torch.fft.fftn(imgs, dim=(2, 3)) * otf
    imgs_conv = torch.real(torch.fft.ifftn(imgs_conv, dim=(2, 3)))
    return imgs_conv
