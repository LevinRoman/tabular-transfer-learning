import dataclasses as dc
import pickle
import typing as ty
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn.preprocessing
import torch
from category_encoders import LeaveOneOutEncoder
from sklearn.impute import SimpleImputer

from . import env, util

ArrayDict = ty.Dict[str, np.ndarray]


def normalize(
    X: ArrayDict, normalization: str, seed: int, noise: float = 1e-3
) -> ArrayDict:
    X_train = X['train'].copy()
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X['train'].shape[0] // 30, 1000), 10),
            subsample=1e9,
            random_state=seed,
        )
        if noise:
            stds = np.std(X_train, axis=0, keepdims=True)
            noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
            X_train += noise_std * np.random.default_rng(seed).standard_normal(  # type: ignore[code]
                X_train.shape
            )
    else:
        util.raise_unknown('normalization', normalization)
    normalizer.fit(X_train)
    return {k: normalizer.transform(v) for k, v in X.items()}  # type: ignore[code]


@dc.dataclass
class Dataset:
    N: ty.Optional[ArrayDict]
    C: ty.Optional[ArrayDict]
    y: ArrayDict
    info: ty.Dict[str, ty.Any]

    @property
    def is_binclass(self) -> bool:
        return self.info['task_type'] == util.BINCLASS

    @property
    def is_multiclass(self) -> bool:
        return self.info['task_type'] == util.MULTICLASS

    @property
    def is_regression(self) -> bool:
        return self.info['task_type'] == util.REGRESSION

    @property
    def n_num_features(self) -> int:
        return self.info['n_num_features']

    @property
    def n_cat_features(self) -> int:
        return self.info['n_cat_features']

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    def size(self, part: str) -> int:
        X = self.N if self.N is not None else self.C
        assert X is not None
        return len(X[part])

    def build_X(
        self,
        *,
        normalization: ty.Optional[str],
        num_nan_policy: str,
        cat_nan_policy: str,
        cat_policy: str,
        cat_min_frequency: float = 0.0,
        seed: int,
    ) -> ty.Union[ArrayDict, ty.Tuple[ArrayDict, ArrayDict]]:

        print('Building Dataset')
        # Nmerical features, replacing NaNs
        if self.N:
            N = deepcopy(self.N)
            num_nan_masks_int = {k: (~np.isnan(v)).astype(int) for k, v in N.items()}
            num_nan_masks = {k: np.isnan(v) for k, v in N.items()}

            if any(x.any() for x in num_nan_masks.values()):  # type: ignore[code]
                if num_nan_policy == 'mean':
                    num_new_values = np.nanmean(self.N['train'], axis=0)
                else:
                    util.raise_unknown('numerical NaN policy', num_nan_policy)
                for k, v in N.items():
                    num_nan_indices = np.where(num_nan_masks[k])
                    v[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])
            # Applying normalization
            if normalization:
                N = normalize(N, normalization, seed)

        else:
            N = None
            num_nan_masks_int = None


        # if there are no categorical features, return only numerical features
        if cat_policy == 'drop' or not self.C:
            assert N is not None
            return N


        # if there are cat features, pre-process them
        C = deepcopy(self.C)
        # replacing missing values
        print(np.array(C['train'][0].dtype))
        cat_nan_masks_int = {k: (v!='MissingValue').astype(int) for k, v in C.items()}
        cat_nan_masks = {k: v == 'MissingValue' for k, v in C.items()}


        if any(x.any() for x in cat_nan_masks.values()):  # type: ignore[code]
            if cat_nan_policy == 'new':
                cat_new_value = '___null___'
                imputer = None
            elif cat_nan_policy == 'most_frequent':
                cat_new_value = None
                imputer = SimpleImputer(strategy=cat_nan_policy)  # type: ignore[code]
                imputer.fit(C['train'])
            else:
                util.raise_unknown('categorical NaN policy', cat_nan_policy)
            if imputer:
                C = {k: imputer.transform(v) for k, v in C.items()}
            else:
                for k, v in C.items():
                    cat_nan_indices = np.where(cat_nan_masks[k])
                    v[cat_nan_indices] = cat_new_value
        # if there is minimum frequency for categorical features specify, replace all rate values with __rare__
        if cat_min_frequency:
            C = ty.cast(ArrayDict, C)
            min_count = round(len(C['train']) * cat_min_frequency)
            rare_value = '___rare___'
            C_new = {x: [] for x in C}
            for column_idx in range(C['train'].shape[1]):
                counter = Counter(C['train'][:, column_idx].tolist())
                popular_categories = {k for k, v in counter.items() if v >= min_count}
                for part in C_new:
                    C_new[part].append(
                        [
                            (x if x in popular_categories else rare_value)
                            for x in C[part][:, column_idx].tolist()
                        ]
                    )
            C = {k: np.array(v).T for k, v in C_new.items()}

        # Encode categorical features as an integer array.
        unknown_value = np.iinfo('int64').max - 3
        encoder = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',  # type: ignore[code]
            unknown_value=unknown_value,  # type: ignore[code]
            dtype='int64',  # type: ignore[code]
        ).fit(C['train'])
        C = {k: encoder.transform(v) for k, v in C.items()}
        max_values = C['train'].max(axis=0)
        for part in ['val', 'test']:
            for column_idx in range(C[part].shape[1]):
                C[part][C[part][:, column_idx] == unknown_value, column_idx] = (
                    max_values[column_idx] + 1
                )

        # choose how to store cat features (?)
        if cat_policy == 'indices':
            result = (N, C, num_nan_masks_int, cat_nan_masks_int)

        elif cat_policy == 'ohe':
            ohe = sklearn.preprocessing.OneHotEncoder(
                handle_unknown='ignore', sparse=False, dtype='float32'  # type: ignore[code]
            )
            ohe.fit(C['train'])
            C = {k: ohe.transform(v) for k, v in C.items()}
            result = C if N is None else {x: np.hstack((N[x], C[x])) for x in N}
            print(result['train'])
        elif cat_policy == 'counter':
            assert seed is not None
            loo = LeaveOneOutEncoder(sigma=0.1, random_state=seed, return_df=False)
            loo.fit(C['train'], self.y['train'])
            C = {k: loo.transform(v).astype('float32') for k, v in C.items()}  # type: ignore[code]
            if not isinstance(C['train'], np.ndarray):
                C = {k: v.values for k, v in C.items()}  # type: ignore[code]
            if normalization:
                C = normalize(C, normalization, seed, inplace=True)  # type: ignore[code]
            result = C if N is None else {x: np.hstack((N[x], C[x])) for x in N}
        else:
            util.raise_unknown('categorical policy', cat_policy)

        return result  # type: ignore[code]

    def build_y(
        self, policy: ty.Optional[str]
    ) -> ty.Tuple[ArrayDict, ty.Optional[ty.Dict[str, ty.Any]]]:
        if self.is_regression:
            assert policy == 'mean_std'
        y = deepcopy(self.y)
        if policy:
            if not self.is_regression:
                warnings.warn('y_policy is not None, but the task is NOT regression')
                info = None
            elif policy == 'mean_std':
                mean, std = self.y['train'].mean(), self.y['train'].std()
                y = {k: (v - mean) / std for k, v in y.items()}
                info = {'policy': policy, 'mean': mean, 'std': std}
            else:
                util.raise_unknown('y policy', policy)
        else:
            info = None
        return y, info


def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:

    return {k: torch.as_tensor(v) for k, v in data.items()}


def load_dataset_info(dataset_name: str) -> ty.Dict[str, ty.Any]:
    info = util.load_json(env.DATA_DIR / dataset_name / 'info.json')
    info['size'] = info['train_size'] + info['val_size'] + info['test_size']
    return info
