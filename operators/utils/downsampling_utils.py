import torch
import math

def linear_filter(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


def cubic_filter_a5(x):
    absx = abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2))


def cubic_filter_a75(x):
    absx = abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.25 * absx3 - 2.25 * absx2 + 1) * (absx <= 1) + (-0.75 * absx3 + 3.75 * absx2 - 6 * absx + 3) * ((1 < absx) & (absx <= 2))


def lanczos2_filter(x):
    epsilon = torch.finfo(torch.float64).eps
    f = (torch.sin(math.pi * x) * torch.sin(math.pi * x / 2) + epsilon) / ((math.pi**2 * x**2 / 2) + epsilon)
    return f * (torch.abs(x) < 2)


def lanczos3_filter(x):
    epsilon = torch.finfo(torch.float64).eps
    f = (torch.sin(math.pi * x) * torch.sin(math.pi * x / 3) + epsilon) / ((math.pi**2 * x**2 / 3) + epsilon)
    return f * (torch.abs(x) < 3)

#Gaussian filter with zeros beyond 4*sigma
def gauss4s_filter(x,sig=0):
    f = torch.exp(-(.5*(x/sig)**2))
    return f * (torch.abs(x) < 4*sig)

#Gaussian filter with zeros beyond 3*sigma
def gauss3s_filter(x,sig=0):
    f = torch.exp(-(.5*(x/sig)**2))
    return f * (torch.abs(x) < 3*sig)

def gen_dwn_kernel1D(dwn_factor, filter=linear_filter, filter_width=None, filter_param=None, antialias=True, cross_correlation=True, device=None):
    assert int(dwn_factor) == dwn_factor, 'downsampling factor must be integer'

    if filter == linear_filter:
        assert filter_param is None, 'The linear filter does not require parameters. The argument \'filter_param\' should be left empty.'
        filter_width = 2.0
    elif filter == cubic_filter_a5 or filter == cubic_filter_a75 or filter == lanczos2_filter:
        assert filter_param is None, 'The cubic or lanczos2 filter does not require parameters. The argument \'filter_param\' should be left empty.'
        filter_width = 4.0
    elif filter == lanczos3_filter:
        assert filter_param is None, 'The cubic or lanczos3 filter does not require parameters. The argument \'filter_param\' should be left empty.'
        filter_width = 6.0
    elif filter == gauss4s_filter:
        assert filter_param is not None, 'filter parameter (sigma) should be specified for the Gaussian filter.'
        filter_width = 2 * math.ceil(4 * filter_param)
    elif filter == gauss3s_filter:
        assert filter_param is not None, 'filter parameter (sigma) should be specified for the Gaussian filter.'
        filter_width = 2 * math.ceil(3 * filter_param)
    elif filter_width is None:
        raise Exception("filter width should be specified for the given custom filter.")


    if antialias:
        filter_width *= dwn_factor
        if filter_param is None:
            filter_ = lambda x: filter(x / dwn_factor) / dwn_factor
        else:
            filter_ = lambda x : filter(x / dwn_factor, filter_param) / dwn_factor

    else:
        if filter_param is None:
            filter_ = filter
        else:
            filter_ = lambda x: filter(x, filter_param)


    u = 0.5 * (1 - dwn_factor)
    right = torch.tensor(u - math.floor(u - filter_width / 2))
    left = right - math.ceil(filter_width) - 1
    kernel_width = 1 + (right-left).item()

    while filter_(left) == 0 and left < 0:
        left += 1
        kernel_width -= 1
    while filter_(right) == 0 and right >= 0:
        right -= 1
        kernel_width -= 1

    #compute discrete kernel from the filter
    if cross_correlation:
        start = right
        stop = left -1
        dir = -1
    else:
        start = left
        stop = right + 1
        dir = 1
    kernel1D = filter_(torch.arange(start,stop,dir,device=device))

    #Normalize kernel
    kernel1D /= torch.sum(kernel1D)

    # Compute kernel center : shifts the kernel by (dwn_factor - 1) // 2 to compensate for the shift in decimation step.
    kernel_center_id = kernel_width // 2
    kernel_center_id -= dir<0 and (kernel_width+1)%2  #correction of kernel center in the case of even width and inverted kernel (i.e. cross correlation mode).
    kernel_center_id += dir * ((dwn_factor - 1) // 2) #shifts the kernel by (dwn_factor - 1) // 2 to compensate for the shift in decimation step.

    return kernel1D, int(kernel_center_id)


def gen_dwn_kernel2D(dwn_factor, filter=linear_filter, filter_width=None, filter_param=None, antialias=True, cross_correlation=True, device=None):
    kernel1D, kernel_center_id = gen_dwn_kernel1D(dwn_factor, filter, filter_width, filter_param, antialias, cross_correlation, device)
    return kernel1D.view(-1,1) @ kernel1D.view(1,-1), kernel_center_id
