""" mimic_tools.py
    Utilities for splitting MetaMIMIC data into upstream and downstream tasks
    Developed for Tabular-Transfer-Learning project
    March 2022
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
import pandas as pd
import torch
import sklearn
from sklearn.model_selection import train_test_split

def split_mimic():
    mimic = pd.read_csv('../../../data/mimic/metaMIMIC.csv', delimiter = ',')
    mimic_target_columns = ['diabetes_diagnosed', 'hypertensive_diagnosed', 'ischematic_diagnosed',
                              'heart_diagnosed', 'overweight_diagnosed', 'anemia_diagnosed', 'respiratory_diagnosed',
                              'hypotension_diagnosed', 'lipoid_diagnosed', 'atrial_diagnosed', 'purpura_diagnosed', 'alcohol_diagnosed']
    y_full = mimic[mimic_target_columns]
    mimic.drop(columns = ['subject_id'], inplace = True)
    mimic.drop(columns = mimic_target_columns, inplace = True)
    X_full = mimic.astype('float')
    categorical_columns = ['gender']
    numerical_columns = list(X_full.columns[X_full.columns != 'gender'])
    X_full.loc[X_full['gender'] == 1, 'gender'] = 'male'
    X_full.loc[X_full['gender'] == 0, 'gender'] = 'female'


    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1875, random_state=1) # 0.1875 x 0.8 = 0.15

    X_train.to_csv('../../../data/mimic/mimic_train_X.csv', index = False)
    X_val.to_csv('../../../data/mimic/mimic_val_X.csv', index = False)
    X_test.to_csv('../../../data/mimic/mimic_test_X.csv', index = False)
    y_train.to_csv('../../../data/mimic/mimic_train_y.csv', index = False)
    y_val.to_csv('../../../data/mimic/mimic_val_y.csv', index = False)
    y_test.to_csv('../../../data/mimic/mimic_test_y.csv', index = False)
    return



def data_prep_transfer_mimic(ds_id, task, stage='pretrain', downstream_target=0, downstream_samples_per_class = 2):
    """
    Function to create a transfer learning task based on the metaMIMIC data
    """
    seed = 0
    np.random.seed(seed)

    mimic_target_columns = ['diabetes_diagnosed', 'hypertensive_diagnosed', 'ischematic_diagnosed',
                            'heart_diagnosed', 'overweight_diagnosed', 'anemia_diagnosed', 'respiratory_diagnosed',
                            'hypotension_diagnosed', 'lipoid_diagnosed', 'atrial_diagnosed', 'purpura_diagnosed',
                            'alcohol_diagnosed']
    X_train = pd.read_csv('../../../data/mimic/mimic_train_X.csv')
    X_val = pd.read_csv('../../../data/mimic/mimic_val_X.csv')
    X_test = pd.read_csv('../../../data/mimic/mimic_test_X.csv')
    y_train_full = pd.read_csv('../../../data/mimic/mimic_train_y.csv')
    y_val_full = pd.read_csv('../../../data/mimic/mimic_val_y.csv')
    y_test_full = pd.read_csv('../../../data/mimic/mimic_test_y.csv')
    categorical_columns = ['gender']
    numerical_columns = list(X_train.columns[X_train.columns != 'gender'])
    X_train[categorical_columns] = X_train[categorical_columns].fillna("MissingValue")
    X_val[categorical_columns] = X_val[categorical_columns].fillna("MissingValue")
    X_test[categorical_columns] = X_test[categorical_columns].fillna("MissingValue")
    print(numerical_columns)
    print(categorical_columns)

    if task == 'binclass':
        if 'downstream' in stage:
            #Merge validation set into train, keep the dummy validation set for the code not to fail
            y_train_full = pd.concat([y_train_full, y_val_full], ignore_index=True)
            X_train = pd.concat([X_train, X_val], ignore_index=True)
            print('Using downstream target:', mimic_target_columns[downstream_target])
            y_train = y_train_full[mimic_target_columns[downstream_target]]
            y_val = y_val_full[mimic_target_columns[downstream_target]]
            y_test = y_test_full[mimic_target_columns[downstream_target]]
        elif 'pretrain' in stage:
            #Do multitarget in regular pretrain
            print('Dropping downstream target:', mimic_target_columns[downstream_target])
            y_train = y_train_full.drop(columns=[mimic_target_columns[downstream_target]])
            y_val = y_val_full.drop(columns=[mimic_target_columns[downstream_target]])
            y_test = y_test_full.drop(columns=[mimic_target_columns[downstream_target]])
        else:
            raise ValueError('Stage is incorrect!')
    else:
        raise NotImplementedError('Mimic only accepts binclass tasks: binclass with multiple targets for pretraining and binclass with a single target for downstream')

    X_train_full = X_train.copy()
    y_train_full = y_train.copy()
    if ('downstream' in stage):
        #switching to downstream_samples_per_class
        print('Total num classes:', len(set(y_train)))
        total_num_of_classes = len(set(y_train))
        X_train, _, y_train, _ = train_test_split(X_train, y_train,
                                       train_size=downstream_samples_per_class * len(set(y_train)),
                                       stratify=y_train, random_state = seed)
        print('Sample num classes:', len(set(y_train)))
        sample_num_classes = len(set(y_train))
        if sample_num_classes < total_num_of_classes:
            print('Resampling and guaranteeing at least one sample per class')
            X_train, y_train = stratified_sample_at_least_one_per_class(X_train_full, y_train_full, downstream_samples_per_class, seed)
            sample_num_classes = len(set(y_train))
            print('New sample num classes:', len(set(y_train)))
        assert total_num_of_classes == sample_num_classes


    X_cat_train = X_train[categorical_columns].values
    X_num_train = X_train[numerical_columns].values
    y_train = y_train.values.astype('float')

    X_cat_val = X_val[categorical_columns].values
    X_num_val = X_val[numerical_columns].values
    y_val = y_val.values.astype('float')

    X_cat_test = X_test[categorical_columns].values
    X_num_test = X_test[numerical_columns].values
    y_test = y_test.values.astype('float')

    info = {}
    info['name'] = ds_id
    info['stage'] = stage
    info['split'] = seed
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
        #We should not have access to a validation set in the limited data regime, replace it with train to make sure
        if ('downstream' in stage):
            X_num_val = X_num_train
        numerical_data = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test}
    else:
        numerical_data = None

    if len(categorical_columns) > 0:
        #We should not have access to a validation set in the limited data regime, replace it with train to make sure
        if ('downstream' in stage):
            X_cat_val = X_cat_train
        categorical_data = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test}
    else:
        categorical_data = None

    #We should not have access to a validation set in the limited data regime, replace it with train to make sure
    if ('downstream' in stage):
        y_val = y_train
    targets = {'train': y_train, 'val': y_val, 'test': y_test}
    print('\n Train size:{} Val size:{} Test size:{}'.format(len(y_train), len(y_val), len(y_test)))

    if len(categorical_columns) > 0:
        #this only works with mimic since the only categorical feature is gender
        full_cat_data_for_encoder = X_train_full[categorical_columns]
    else:
        full_cat_data_for_encoder = None

    return numerical_data, categorical_data, targets, info, full_cat_data_for_encoder


def stratified_sample_at_least_one_per_class(X_train, y_train, downstream_samples_per_class, seed):
    # Sample 1 element per class
    X_train['y'] = y_train
    X_one_sample = X_train.groupby(by='y').sample(n=1)
    y_one_sample = X_one_sample['y']
    X_one_sample = X_one_sample.drop(columns=['y'])
    # Add a stratified sample from the rest of the data
    X_train = X_train[~X_train.index.isin(X_one_sample.index)]
    y_train = X_train['y']
    X_train = X_train.drop(columns=['y'])
    X_train, _, y_train, _ = train_test_split(X_train, y_train,
                                              train_size=downstream_samples_per_class * len(set(y_train)) - len(
                                                  X_one_sample),
                                              stratify=y_train, random_state=seed)
    X_train = pd.concat([X_train, X_one_sample], axis=0)
    y_train = pd.concat([y_train, y_one_sample], axis=0)
    return X_train, y_train
