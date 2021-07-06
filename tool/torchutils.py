
import torch

from torch.utils.data import Subset
import numpy as np
import math


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def how_many_gpus(cuda=True):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        return torch.cuda.device_count()
    else:
        return 0


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class StepOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, gamma=0.3):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.gamma = gamma

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, epoch, closure=None):
        if (epoch == 0) and (self.global_step < 200):
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = (self.__initial_lr[i] / 200) * (self.global_step + 1)

        else:
            lr_mult = self.gamma ** (np.floor(float(epoch)/3.0))
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step))/2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


def load_state_no_module(path, map_location=torch.device('cpu')):
    state_dict = torch.load(path, map_location=map_location)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] != "module.":
            name = k
        else:
            name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict
