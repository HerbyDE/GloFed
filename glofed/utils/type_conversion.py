import torch

def cast_to_float(x):
    return torch.from_numpy(x).float().cuda() if torch.cuda.is_available() else torch.from_numpy(x).float()

def cast_to_int(x):
    return torch.from_numpy(x).int().cuda() if torch.cuda.is_available() else torch.from_numpy(x).int()