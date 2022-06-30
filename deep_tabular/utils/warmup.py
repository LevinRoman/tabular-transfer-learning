""" warmup.py
    code for warmup learning rate scheduler
    borrowed from https://github.com/ArneNx/pytorch_warmup/tree/warmup_fix
    and modified July 2020
"""

import math

from torch.optim import Optimizer


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class BaseWarmup:
    """Base class for all warmup schedules

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_params (list): warmup paramters
        last_step (int): The index of last step. (Default: -1)
        warmup_period (int or list): Warmup period
    """

    def __init__(self, optimizer, warmup_params, last_step=-1, warmup_period=0):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.warmup_params = warmup_params
        self.last_step = last_step
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.warmup_period = warmup_period
        self.dampen()

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.

        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def dampen(self, step=None):
        """Dampen the learning rates.

        Arguments:
            step (int): The index of current step. (Default: None)
        """
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        if isinstance(self.warmup_period, int) and step < self.warmup_period:
            for i, (group, params) in enumerate(zip(self.optimizer.param_groups,
                                                    self.warmup_params)):
                if isinstance(self.warmup_period, list) and step >= self.warmup_period[i]:
                    continue
                omega = self.warmup_factor(step, **params)
                group['lr'] = omega * self.base_lrs[i]

    def warmup_factor(self, step, warmup_period):
        """Place holder for objects that inherit BaseWarmup."""
        raise NotImplementedError


def get_warmup_params(warmup_period, group_count):
    if type(warmup_period) == list:
        if len(warmup_period) != group_count:
            raise ValueError(
                'size of warmup_period does not equal {}.'.format(group_count))
        for x in warmup_period:
            if type(x) != int:
                raise ValueError(
                    'An element in warmup_period, {}, is not an int.'.format(
                        type(x).__name__))
        warmup_params = [dict(warmup_period=x) for x in warmup_period]
    elif type(warmup_period) == int:
        warmup_params = [dict(warmup_period=warmup_period)
                         for _ in range(group_count)]
    else:
        raise TypeError('{} is not a list nor an int.'.format(
            type(warmup_period).__name__))
    return warmup_params


class LinearWarmup(BaseWarmup):
    """Linear warmup schedule.

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super().__init__(optimizer, warmup_params, last_step, warmup_period)

    def warmup_factor(self, step, warmup_period):
        return min(1.0, (step+1) / warmup_period)


class ExponentialWarmup(BaseWarmup):
    """Exponential warmup schedule.

    Arguments:
        optimizer (Optimizer): an instance of a subclass of Optimizer
        warmup_period (int or list): Effective warmup period
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, warmup_period, last_step=-1):
        group_count = len(optimizer.param_groups)
        warmup_params = get_warmup_params(warmup_period, group_count)
        super().__init__(optimizer, warmup_params, last_step, warmup_period)

    def warmup_factor(self, step, warmup_period):
        if step + 1 >= warmup_period:
            return 1.0
        else:
            return 1.0 - math.exp(-(step+1) / warmup_period)
