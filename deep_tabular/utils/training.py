""" training.py
    Utilities for training models
    Developed for Tabular-Transfer-Learning project
    March 2022
"""

import random
from dataclasses import dataclass
from typing import Any

# from icecream import ic
from tqdm import tqdm


# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114, W0611


@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    criterions: Any
    optimizer: Any
    scheduler: Any
    warmup: Any
    num_datasets_in_batch: Any = None


def default_training_loop(net, trainloader, train_setup, device):
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    criterion = train_setup.criterions

    train_loss = 0
    total = 0

    for batch_idx, (inputs_num, inputs_cat, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
        inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                 inputs_cat if inputs_cat.nelement() != 0 else None

        optimizer.zero_grad()
        outputs = net(inputs_num, inputs_cat)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss

