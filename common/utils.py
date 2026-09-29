import hashlib
import os

import numpy as np
import torch


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            self.sum = np.zeros(val.shape)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_ckpt(state, ckpt_path, suffix=None):
    if suffix is None:
        suffix = 'epoch_{:04d}'.format(state['epoch'])
    torch.save(state, os.path.join(ckpt_path, 'ckpt_{}.pth.tar'.format(suffix)))


def wrap(func, unsqueeze, *args):
    """Call a torch function with NumPy arrays; inputs and outputs are converted transparently."""
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    return result
