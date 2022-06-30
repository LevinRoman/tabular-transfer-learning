""" testing.py
    Utilities for testing models
    Developed for Tabular-Transfer-Learning project
    March 2022
    Some functionality adopted from https://github.com/Yura52/rtdl
"""

import torch
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def evaluate_model(net, loaders, task, device):
    scores = []
    for loader in loaders:
        score = test_default(net, loader, task, device)
        scores.append(score)
    return scores


def test_default(net, testloader, task, device):
    net.eval()
    targets_all = []
    predictions_all = []
    with torch.no_grad():
        for batch_idx, (inputs_num, inputs_cat, targets) in enumerate(tqdm(testloader, leave=False)):
            inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
            inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                     inputs_cat if inputs_cat.nelement() != 0 else None

            outputs = net(inputs_num, inputs_cat)
            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = outputs
            elif task == "regression":
                predicted = outputs
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predicted.cpu().tolist())

    if task == "multiclass":
        accuracy = accuracy_score(targets_all, predictions_all)
        balanced_accuracy = balanced_accuracy_score(targets_all, predictions_all, adjusted=False)
        balanced_accuracy_adjusted = balanced_accuracy_score(targets_all, predictions_all, adjusted=True)
        scores = {"score": accuracy,
                  "accuracy": accuracy,
                  "balanced_accuracy": balanced_accuracy,
                  "balanced_accuracy_adjusted": balanced_accuracy_adjusted}
    elif task == "regression":
        rmse = mean_squared_error(targets_all, predictions_all, squared=False)
        scores = {"score": -rmse,
                  "rmse": -rmse}
    elif task == "binclass":
        roc_auc = roc_auc_score(targets_all, predictions_all)
        scores = {"score": roc_auc,
                  "roc_auc": roc_auc}
    return scores


def evaluate_backbone(embedders, backbone, heads, loaders, tasks, device):
    scores = {}
    for k in loaders.keys():
        score = evaluate_backbone_one_dataset(embedders[k], backbone, heads[k], loaders[k], tasks[k], device)
        scores[k] = score
    return scores


def evaluate_backbone_one_dataset(embedder, backbone, head, testloader, task, device):
    embedder.eval()
    backbone.eval()
    head.eval()
    targets_all = []
    predictions_all = []
    with torch.no_grad():
        for batch_idx, (inputs_num, inputs_cat, targets) in enumerate(tqdm(testloader, leave=False)):
            inputs_num, inputs_cat, targets = inputs_num.to(device).float(), inputs_cat.to(device), targets.to(device)
            inputs_num, inputs_cat = inputs_num if inputs_num.nelement() != 0 else None, \
                                     inputs_cat if inputs_cat.nelement() != 0 else None

            embedding = embedder(inputs_num, inputs_cat)
            features = backbone(embedding)
            outputs = head(features)

            if task == "multiclass":
                predicted = torch.argmax(outputs, dim=1)
            elif task == "binclass":
                predicted = outputs
            elif task == "regression":
                predicted = outputs
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predicted.cpu().tolist())

    if task == "multiclass":
        accuracy = accuracy_score(targets_all, predictions_all)
        balanced_accuracy = balanced_accuracy_score(targets_all, predictions_all, adjusted=False)
        balanced_accuracy_adjusted = balanced_accuracy_score(targets_all, predictions_all, adjusted=True)
        scores = {"score": accuracy,
                  "accuracy": accuracy,
                  "balanced_accuracy": balanced_accuracy,
                  "balanced_accuracy_adjusted": balanced_accuracy_adjusted}
    elif task == "regression":
        rmse = mean_squared_error(targets_all, predictions_all, squared=False)
        scores = {"score": -rmse,
                  "rmse": -rmse}
    elif task == "binclass":
        roc_auc = roc_auc_score(targets_all, predictions_all)
        scores = {"score": roc_auc,
                  "roc_auc": roc_auc}
    return scores
