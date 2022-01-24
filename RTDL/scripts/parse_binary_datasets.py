import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
import pandas as pd
import lib
import argparse

def parse_categories(ds_id):
    #If ft-transformer data
    if type(ds_id) == str:
        cat_path = 'data/{}/C_{}.npy'.format(ds_id, 'train')
        if os.path.exists(cat_path):
            categorical_array = np.vstack([np.load('data/{}/C_{}.npy'.format(ds_id, cur_split)) for cur_split in ['train', 'val', 'test']])
            categorical_columns = ['cat_{}'.format(i) for i in range(categorical_array.shape[1])]
            cat_df = pd.DataFrame(categorical_array, columns=categorical_columns)
        else:
            cat_df = pd.DataFrame()
            categorical_columns = []

        num_path = 'data/{}/N_{}.npy'.format(ds_id, 'train')
        if os.path.exists(num_path):
            numerical_array = np.vstack([np.load('data/{}/N_{}.npy'.format(ds_id, cur_split)) for cur_split in ['train', 'val', 'test']])
            numerical_columns = ['num_{}'.format(i) for i in range(numerical_array.shape[1])]
            num_df =pd.DataFrame(numerical_array, columns=numerical_columns)
        else:
            num_df = pd.DataFrame()
            numerical_columns = []

        X_full = pd.concat([num_df, cat_df], axis = 1)
        assert X_full.shape[0] == num_df.shape[0]

        y_full = np.concatenate([np.load('data/{}/y_{}.npy'.format(ds_id, cur_split)) for cur_split in ['train', 'val', 'test']])
        y_full = pd.Series(y_full, name = 'target')


    #or if SAINT data or generic openml data
    elif type(ds_id) == int:
        dataset = openml.datasets.get_dataset(ds_id)
        # load data
        X_full, y_full, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        # get cat, cont columns
        # categorical_columns, numerical_columns = get_cont_cat_features(ds_id)
        categorical_columns = list(X_full.columns[np.array(categorical_indicator)])
        numerical_columns = list(X_full.columns[~np.array(categorical_indicator)])
    else:
        raise ValueError('Wrong ds_id type. Type of ds_id can only be int or str')
    print(numerical_columns)
    print(categorical_columns)
    print(X_full.shape, y_full.shape, y_full.dtype)
    if y_full.dtype.name == "category":
        y_full = y_full.apply(str).astype('object')

    if len(X_full) > 1000000:#ds_id in [42728, 42705, 42729, 42571]:
        # import ipdb; ipdb.set_trace()
        X_full, y_full = X_full.sample(n=1000000), y_full.sample(n=1000000)
        X_full.reset_index(drop=True, inplace=True)
        y_full.reset_index(drop=True, inplace=True)



    for col in categorical_columns:
        X_full[col] = X_full[col].apply(str).astype("object")

    if len(categorical_columns) > 0:
        full_cat_data_for_encoder = X_full[categorical_columns]
        categories = np.array(lib.get_categories_full_cat_data(full_cat_data_for_encoder))
    else:
        full_cat_data_for_encoder = None
        categories = None
    return categories, X_full.shape[1]

if __name__ == '__main__':
    k = 3 #>=5 binary features
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default=['1483'], type=str, nargs='+')
    # parser.add_argument('--dir_path', default='output/126')
    args = parser.parse_args()

    saint_nonbinary_datasets = [188, 1596, 4541, 40664, 40685, 40687, 40975, 41166, 41169, 42734,
                     541, 42726, 42727, 422, 42571, 42705, 42728, 42563, 42724, 42729]

    ft_transfomer_nonbinary_datasets = ["california_housing", "yahoo", "year", "microsoft", "helena", "aloi", "covtype", "jannis"]
    all_nonbinary_data = saint_nonbinary_datasets + ft_transfomer_nonbinary_datasets
    print('All data ', all_nonbinary_data)
    binary_candidates = []
    for dataset in all_nonbinary_data:
        categories, num_features = parse_categories(dataset)
        if categories is not None:
            num_binary = (categories == 2).sum()
            if num_binary > 1+k:
                binary_candidates.append([dataset, num_binary, num_features])


    ft_transfomer_binary_datasets = ["higgs_small", "adult", "epsilon"]
    saint_binary_datasets = [1487,44,1590,42178,1111,31,42733,1494,1017,4134]

    all_binary_data = ft_transfomer_binary_datasets + saint_binary_datasets
    binary_candidates_from_binary = []
    for dataset in all_binary_data:
        categories, num_features = parse_categories(dataset)
        if categories is not None:
            num_binary = (categories == 2).sum()
            if num_binary > 0+k:
                binary_candidates_from_binary.append([dataset, num_binary, num_features])


    print('\nDone!Total {} binary candidates!'.format(len(binary_candidates)+len(binary_candidates_from_binary)))
    print('\nFrom nonbinary:{}\n'.format(binary_candidates))
    print('\nFrom binary:{}\n'.format(binary_candidates_from_binary))


