""" data_tools.py
    Tools for building tabular datasets
    Developed for Tabular Transfer Learning project
    April 2022
    Some functionality adopted from https://github.com/Yura52/rtdl
"""

import logging
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, Any

# from icecream import ic
import numpy as np
import openml
import pandas as pd
import sklearn.preprocessing
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def get_categories_full_cat_data(full_cat_data_for_encoder):
    return (
        None
        if full_cat_data_for_encoder is None
        else [
            len(set(full_cat_data_for_encoder.values[:, i]))
            for i in range(full_cat_data_for_encoder.shape[1])
        ]
    )


def get_data_openml(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    data, targets, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                             target=dataset.default_target_attribute)
    categorical_columns = list(data.columns[np.array(categorical_indicator)])
    numerical_columns = list(data.columns[~np.array(categorical_indicator)])
    return data, targets, categorical_columns, numerical_columns


def get_data_locally(ds_id):
    if os.path.exists(f'../../../data/{ds_id}/N.csv'):
        X_full_num = pd.read_csv(f'../../../data/{ds_id}/N.csv')
        numerical_columns = list(X_full_num.columns)
    else:
        X_full_num = pd.DataFrame()
        numerical_columns = []
    if os.path.exists(f'../../../data/{ds_id}/C.csv'):
        X_full_cat = pd.read_csv(f'../../../data/{ds_id}/C.csv')
        categorical_columns = list(X_full_cat.columns)
    else:
        X_full_cat = pd.DataFrame()
        categorical_columns = []

    X_full = pd.concat([X_full_num, X_full_cat], axis = 1)
    y_full = pd.read_csv(f'../../../data/{ds_id}/y.csv')

    if y_full.shape[1] == 1:
        y_full = y_full.iloc[:, 0]
    else:
        raise ValueError('Targets have more than one column and the task is not multilabel')


    return X_full, y_full, categorical_columns, numerical_columns

def get_data(dataset_id, source, task, datasplit=[.65, .15, .2]):
    """
    Function to read and prepare a multiclass/binclass/regression dataset
    """
    seed = 0
    np.random.seed(seed)

    if source == 'openml':
        data, targets, categorical_columns, numerical_columns = get_data_openml(dataset_id)
    elif source == 'local':
        data, targets, categorical_columns, numerical_columns = get_data_locally(dataset_id)
        np.random.seed(seed)
    # Fixes some bugs in openml datasets
    if targets.dtype.name == "category":
        targets = targets.apply(str).astype('object')

    for col in categorical_columns:
        data[col] = data[col].apply(str).astype("object")

    # reindex and find NaNs/Missing values in categorical columns
    data, targets = data.reset_index(drop=True), targets.reset_index(drop=True)
    data[categorical_columns] = data[categorical_columns].fillna("___null___")

    if task != 'regression':
        l_enc = LabelEncoder()
        targets = l_enc.fit_transform(targets)
    else:
        targets = targets.to_numpy()

    # split data into train/val/test
    train_size, test_size, valid_size = datasplit[0], datasplit[2], datasplit[1]/(1-datasplit[2])
    if task != 'regression':
        data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=test_size, random_state=seed, stratify = targets)
        data_train, data_val, targets_train, targets_val = train_test_split(data_train, targets_train, test_size=valid_size, random_state=seed, stratify = targets_train)
    else:
        data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=test_size, random_state=seed)
        data_train, data_val, targets_train, targets_val = train_test_split(data_train, targets_train, test_size=valid_size, random_state=seed)



    data_cat_train = data_train[categorical_columns].values
    data_num_train = data_train[numerical_columns].values

    data_cat_val = data_val[categorical_columns].values
    data_num_val = data_val[numerical_columns].values

    data_cat_test = data_test[categorical_columns].values
    data_num_test = data_test[numerical_columns].values

    info = {"name": dataset_id,
            "task_type": task,
            "n_num_features": len(numerical_columns),
            "n_cat_features": len(categorical_columns),
            "train_size": data_train.shape[0],
            "val_size": data_val.shape[0],
            "test_size": data_test.shape[0]}

    if task == "multiclass":
        info["n_classes"] = len(set(targets))
    if task == "binclass":
        info["n_classes"] = 1
    if task == "regression":
        info["n_classes"] = 1

    if len(numerical_columns) > 0:
        numerical_data = {"train": data_num_train, "val": data_num_val, "test": data_num_test}
    else:
        numerical_data = None

    if len(categorical_columns) > 0:
        categorical_data = {"train": data_cat_train, "val": data_cat_val, "test": data_cat_test}
    else:
        categorical_data = None

    targets = {"train": targets_train, "val": targets_val, "test": targets_test}

    if len(categorical_columns) > 0:
        full_cat_data_for_encoder = data[categorical_columns]
    else:
        full_cat_data_for_encoder = None

    return numerical_data, categorical_data, targets, info, full_cat_data_for_encoder


def get_multilabel_data(ds_id, source, task):
    """
    Function to read and prepare a multi-label dataset -- handling of multiple targets is slightly different from the other cases
    """
    if source != 'local':
        raise ValueError("Only locally stored multilabel datasets are accepted. If it is local, double check 'source: local' in dataset config")
    seed = 0
    np.random.seed(seed)
    if os.path.exists(f'../../../data/{ds_id}/N.csv'):
        X_full_num = pd.read_csv(f'../../../data/{ds_id}/N.csv')
        numerical_columns = list(X_full_num.columns)
    else:
        X_full_num = pd.DataFrame()
        numerical_columns = []
    if os.path.exists(f'../../../data/{ds_id}/C.csv'):
        X_full_cat = pd.read_csv(f'../../../data/{ds_id}/C.csv')
        categorical_columns = list(X_full_cat.columns)
    else:
        X_full_cat = pd.DataFrame()
        categorical_columns = []

    X_full = pd.concat([X_full_num, X_full_cat], axis = 1)
    y_full = pd.read_csv(f'../../../data/{ds_id}/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1875, random_state=1)

    X_train[categorical_columns] = X_train[categorical_columns].fillna("MissingValue")
    X_val[categorical_columns] = X_val[categorical_columns].fillna("MissingValue")
    X_test[categorical_columns] = X_test[categorical_columns].fillna("MissingValue")
    # print(numerical_columns)
    # print(categorical_columns)

    X_cat_train = X_train[categorical_columns].values
    X_num_train = X_train[numerical_columns].values.astype('float')
    y_train = y_train.values.astype('float')

    X_cat_val = X_val[categorical_columns].values
    X_num_val = X_val[numerical_columns].values.astype('float')
    y_val = y_val.values.astype('float')

    X_cat_test = X_test[categorical_columns].values
    X_num_test = X_test[numerical_columns].values.astype('float')
    y_test = y_test.values.astype('float')

    info = {}
    info['name'] = ds_id
    info['task_type'] = task
    info['n_num_features'] = len(numerical_columns)
    info['n_cat_features'] = len(categorical_columns)
    info['train_size'] = X_train.shape[0]
    info['val_size'] = X_val.shape[0]
    info['test_size'] = X_test.shape[0]


    if len(y_train.shape) > 1:
        info['n_classes'] = y_train.shape[1]
    else:
        info['n_classes'] = 1

    if len(numerical_columns) > 0:
        numerical_data = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test}
    else:
        numerical_data = None

    if len(categorical_columns) > 0:
        categorical_data = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test}
    else:
        categorical_data = None

    targets = {'train': y_train, 'val': y_val, 'test': y_test}
    print('\n Train size:{} Val size:{} Test size:{}'.format(len(y_train), len(y_val), len(y_test)))

    if len(categorical_columns) > 0:
        full_cat_data_for_encoder = X_full[categorical_columns]
    else:
        full_cat_data_for_encoder = None
    return numerical_data, categorical_data, targets, info, full_cat_data_for_encoder

@dataclass
class TabularDataset:
    x_num: Optional[Dict[str, np.ndarray]]
    x_cat: Optional[Dict[str, np.ndarray]]
    y: Dict[str, np.ndarray]
    info: Dict[str, Any]
    normalization: Optional[str]
    cat_policy: str
    seed: int
    full_cat_data_for_encoder: Optional[pd.DataFrame]
    y_policy: Optional[str] = None
    normalizer_path: Optional[str] = None
    stage: Optional[str] = None

    @property
    def is_binclass(self):
        return self.info['task_type'] == "binclass"

    @property
    def is_multiclass(self):
        return self.info['task_type'] == "multiclass"

    @property
    def is_regression(self):
        return self.info['task_type'] == "regression"

    @property
    def n_num_features(self):
        return self.info["n_num_features"]

    @property
    def n_cat_features(self):
        return self.info["n_cat_features"]

    @property
    def n_features(self):
        return self.n_num_features + self.n_cat_features

    @property
    def n_classes(self):
        return self.info["n_classes"]

    @property
    def parts(self):
        return self.x_num.keys() if self.x_num is not None else self.x_cat.keys()

    def size(self, part: str):
        x = self.x_num if self.x_num is not None else self.x_cat
        assert x is not None
        return len(x[part])

    def normalize(self, x_num, noise=1e-3):
        x_num_train = x_num['train'].copy()
        if self.normalization == 'standard':
            normalizer = sklearn.preprocessing.StandardScaler()
        elif self.normalization == 'quantile':
            normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(x_num['train'].shape[0] // 30, 1000), 10),
                subsample=1e9,
                random_state=self.seed,
            )
            if noise:
                stds = np.std(x_num_train, axis=0, keepdims=True)
                noise_std = noise / np.maximum(stds, noise)
                x_num_train += noise_std * np.random.default_rng(self.seed).standard_normal(x_num_train.shape)
        else:
            raise ValueError('Unknown Normalization')
        normalizer.fit(x_num_train)
        if self.normalizer_path is not None:
            if self.stage is None:
                raise ValueError('stage is None, only pretrain or downstream are accepted if normalizer_path is not None')
            if self.stage == 'pretrain':
                pickle.dump(normalizer, open(self.normalizer_path, 'wb'))
                print(f'Normalizer saved to {self.normalizer_path}')
            if self.stage == 'downstream':
                normalizer = pickle.load(open(self.normalizer_path, 'rb'))
                print(f'Normalizer loaded from {self.normalizer_path}')
        return {k: normalizer.transform(v) for k, v in x_num.items()}

    def handle_missing_values_numerical_features(self, x_num):
        # TODO: handle num_nan_masks for SAINT
        # num_nan_masks_int = {k: (~np.isnan(v)).astype(int) for k, v in x_num.items()}
        num_nan_masks = {k: np.isnan(v) for k, v in x_num.items()}
        if any(x.any() for x in num_nan_masks.values()):

            # TODO check if we need self.x_num here
            num_new_values = np.nanmean(self.x_num['train'], axis=0)
            for k, v in x_num.items():
                num_nan_indices = np.where(num_nan_masks[k])
                v[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])
        return x_num

    def encode_categorical_features(self, x_cat):
        encoder = sklearn.preprocessing.OrdinalEncoder(handle_unknown='error', dtype='int64')
        encoder.fit(self.full_cat_data_for_encoder.values)
        x_cat = {k: encoder.transform(v) for k, v in x_cat.items()}
        return x_cat

    def transform_categorical_features_to_ohe(self, x_cat):
        ohe = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False, dtype='float32')
        ohe.fit(self.full_cat_data_for_encoder.astype('str'))
        x_cat = {k: ohe.transform(v.astype('str')) for k, v in x_cat.items()}
        return x_cat

    def concatenate_data(self, x_cat, x_num):
        if self.cat_policy == 'indices':
            result = [x_num, x_cat]

        elif self.cat_policy == 'ohe':
            # TODO: handle output for models that need ohe
            raise ValueError('Not implemented')
        return result

    def preprocess_data(self):
        # TODO: seed (?)
        logging.info('Building Dataset')
        # TODO: figure out if we really need a copy of data or if we can preprocess it in place
        if self.x_num:
            x_num = deepcopy(self.x_num)
            x_num = self.handle_missing_values_numerical_features(x_num)
            if self.normalization:
                x_num = self.normalize(x_num)
        else:
            # if x_num is None replace with empty tensor for dataloader
            x_num = {part: torch.empty(self.size(part), 0) for part in self.parts}

        # if there are no categorical features, return only numerical features
        if self.cat_policy == 'drop' or not self.x_cat:
            assert x_num is not None
            x_num = to_tensors(x_num)
            # if x_cat is None replace with empty tensor for dataloader
            x_cat = {part: torch.empty(self.size(part), 0) for part in self.parts}
            return [x_num, x_cat]

        x_cat = deepcopy(self.x_cat)
        # x_cat_nan_masks = {k: v == '___null___' for k, v in x_cat.items()}
        x_cat = self.encode_categorical_features(x_cat)

        x_cat, x_num = to_tensors(x_cat), to_tensors(x_num)
        result = self.concatenate_data(x_cat, x_num)

        return result

    def build_y(self):
        if self.is_regression:
            assert self.y_policy == 'mean_std'
        y = deepcopy(self.y)
        if self.y_policy:
            if not self.is_regression:
                warnings.warn('y_policy is not None, but the task is NOT regression')
                info = None
            elif self.y_policy == 'mean_std':
                mean, std = self.y['train'].mean(), self.y['train'].std()
                y = {k: (v - mean) / std for k, v in y.items()}
                info = {'policy': self.y_policy, 'mean': mean, 'std': std}
            else:
                raise ValueError('Unknown y policy')
        else:
            info = None

        y = to_tensors(y)
        if self.is_regression or self.is_binclass:
            y = {part: y[part].float() for part in self.parts}
        return y, info


def to_tensors(data):
    return {k: torch.as_tensor(v) for k, v in data.items()}
