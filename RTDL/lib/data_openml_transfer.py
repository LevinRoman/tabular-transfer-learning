'''
ADDED BY MICAH FOR TRANSFER LEARNING PROJECT.  ONLY SUPPORTS MULTI-CLASS CLASSIFICATION RIGHT NOW
'''
import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.utils.data import Dataset
import random
import os
import pickle

def simple_lapsed_time(text, lapsed):
    hours, rem = divmod(lapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(text + ": {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def task_dset_ids(task):
    dataset_ids = {
        'binary': [1487, 44, 1590, 42178, 1111, 31, 42733, 1494, 1017, 4134],
        'multiclass': [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734],
        # Start with 1596, 40687, 40975, 41166, 41169. Could consider 1110 or 1113 since they have lots of data?
        'regression': [541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]
    }

    return dataset_ids[task]


def concat_data(X, y):
    # import ipdb; ipdb.set_trace()
    return pd.concat([pd.DataFrame(X['data']), pd.DataFrame(y['data'][:, 0].tolist(), columns=['target'])], axis=1)


def data_split(X, y, nan_mask, indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise 'Shape of data not same as that of nan mask!'

    y_d = {
        'data': y[indices].reshape(-1, 1)
    }
    return x_d, y_d


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):

        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns
        if task == 'clf':
            self.y = Y['data']  # .astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.y[idx], np.concatenate(
            (self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx]


def class_split(dataset, ds_id, p=0.5, class_split_path=os.path.join(os.getcwd(), "class_splits/")):
    '''
    splits classes into pretraining classes with proportion p and downstream classes with proportion 1-p,
    rounds down the number of pretraining classes
    '''
    os.makedirs(class_split_path, exist_ok=True)
    pickle_path = os.path.join(class_split_path, str(ds_id) + '_classes.pickle')

    if os.path.isfile(pickle_path):
        # Load data (deserialize)
        with open(pickle_path, 'rb') as handle:
            classes = pickle.load(handle)
    else:
        shuffled = random.sample(dataset.retrieve_class_labels(), len(dataset.retrieve_class_labels()))
        classes = {'pretrain': shuffled[:int(p * len(shuffled))], 'downstream': shuffled[int(p * len(shuffled)):]}
        # Store data (serialize)
        with open(pickle_path, 'wb') as handle:
            pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return classes


def split_indices_by_class(X, y, keep_classes):
    return y.isin(keep_classes)


def data_prep_openml_transfer(ds_id, seed, task, stage='pretrain', datasplit=[.65, .15, .2], samples=False,
                              pretrain_proportion=0.5):
    random.seed(seed)
    dataset = openml.datasets.get_dataset(ds_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                    target=dataset.default_target_attribute)

    classes = class_split(dataset, ds_id, p=pretrain_proportion)
    # X, y = split_data_by_class(X, y, classes[stage])
    indices = split_indices_by_class(X, y, classes[stage])

    if samples:
        X = X.sample(samples)
        y = y[X.index]
        X.reset_index(inplace=True, drop=True), y.reset_index(inplace=True, drop=True)

    np.random.seed(seed)

    if ds_id == 42178:
        categorical_indicator = [True, False, True, True, False, True, True, True, True, True, True, True, True, True,
                                 True, True, True, False, False]
        tmp = [x if (x != ' ') else '0' for x in X['TotalCharges'].tolist()]
        X['TotalCharges'] = [float(i) for i in tmp]
        y = y[X.TotalCharges != 0]
        X = X[X.TotalCharges != 0]
        X.reset_index(drop=True, inplace=True)
        print(y.shape, X.shape)
    if ds_id in [42728, 42705, 42729, 42571]:
        # import ipdb; ipdb.set_trace()
        X, y = X[:50000], y[:50000]
        X.reset_index(drop=True, inplace=True)
    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator) == True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)

    cat_dims = []
    for col in categorical_columns:
        #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
        #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    if task != 'regression':
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)

    X, y, nan_mask = X[indices].reset_index(drop=True), y[indices], nan_mask[indices].reset_index(
        drop=True)  # do i need to reset indices of nan_mask

    if samples:
        X = X.sample(samples)
        y = y[X.index]
        nan_mask = nan_mask[X.index]
        X.reset_index(inplace=True, drop=True), y.reset_index(inplace=True, drop=True), nan_mask.reset_index(
            inplace=True, drop=True)  # do i need to reset indices of nan_mask

    unique = np.unique(y)
    for val in range(len(unique)):
        y = np.where(y == unique[val], val, y)

    X["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))

    train_indices = X[X.Set == "train"].index
    valid_indices = X[X.Set == "valid"].index
    test_indices = X[X.Set == "test"].index

    X = X.drop(columns=['Set'])

    X_train, y_train = data_split(X, y, nan_mask, train_indices)
    X_valid, y_valid = data_split(X, y, nan_mask, valid_indices)
    X_test, y_test = data_split(X, y, nan_mask, test_indices)

    train_mean, train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0), np.array(
        X_train['data'][:, con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std



