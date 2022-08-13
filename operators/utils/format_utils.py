import torch


#Convert arbitrary iterable type (e.g. list, tuple, numpy array) or scalar value to a tensor (keep input dimensions).
def to_tensor(numeric, device=None):
    if not isinstance(numeric, torch.Tensor):
        return torch.tensor(numeric if hasattr(numeric, '__getitem__') else [numeric], device=device)
    else:
        return numeric.to(device)
