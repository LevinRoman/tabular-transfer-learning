import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
import pandas as pd
import lib
import torch
import sklearn
from sklearn.model_selection import train_test_split

def class_split(y, ds_id, p=0.5, class_split_path=os.path.join('transfer', "class_splits/")):
    '''
    splits classes into pretraining classes with proportion p and downstream classes with proportion 1-p,
    rounds down the number of pretraining classes
    '''
    random.seed(0)
    labels = set(y)
    print('Total number of classes: {}\n'.format(len(labels)))
    os.makedirs(class_split_path, exist_ok=True)
    pickle_path = os.path.join(class_split_path, '{}_p{}_class_split.pickle'.format(ds_id, p))

    if os.path.isfile(pickle_path):
        # Load data (deserialize)
        with open(pickle_path, 'rb') as handle:
            classes = pickle.load(handle)
    else:
        shuffled = random.sample(labels, len(labels))
        classes = {'pretrain': shuffled[:int(p * len(shuffled))],
                   'downstream': shuffled[int(p * len(shuffled)):],
                   'pretrain_proportion': p}
        # Store data (serialize)
        with open(pickle_path, 'wb') as handle:
            pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return classes

def regression_split(y, ds_id, q=0.5, regression_split_path=os.path.join('transfer', 'regression_splits/'), direction = 'downstream_ge_q'):
    print('{}-th quantile: {}\n'.format(q, np.quantile(y, q)))
    os.makedirs(regression_split_path, exist_ok=True)
    pickle_path = os.path.join(regression_split_path, '{}_q{}_regression_split.pickle'.format(ds_id, q))

    if os.path.isfile(pickle_path):
        # Load data (deserialize)
        with open(pickle_path, 'rb') as handle:
            regression_split_dict = pickle.load(handle)
    else:
        regression_split_dict = {'direction': direction,
                   'quantile': q,
                   'quantile_value': np.quantile(y, q)}
        # Store data (serialize)
        with open(pickle_path, 'wb') as handle:
            pickle.dump(regression_split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return regression_split_dict

def split_indices_by_class(X, y, keep_classes):
    return y.isin(keep_classes)

def regression_split_indices(X, y, regression_split, stage):
    if regression_split['direction'] == 'downstream_ge_q':
        if 'downstream' in stage:
            return y >= regression_split['quantile_value']
        elif 'pretrain' in stage:
            return y < regression_split['quantile_value']
        else:
            raise ValueError('Stage can only be downstream or pretrain')
    elif regression_split['direction'] == 'downstream_le_q':
        if 'downstream' in stage:
            return y <= regression_split['quantile_value']
        elif 'pretrain' in stage:
            return y > regression_split['quantile_value']
        else:
            raise ValueError('Stage can only be downstream or pretrain')
    else:
        raise NotImplementedError('The only implemented splits are downstream_ge_q and downstream_le_q')
    return

def binary_multilabel_targets(ds_id, X_full, categorical_columns, n = 5, multilabel_split_path = os.path.join('transfer', 'multilabel_splits/')):
    """Randomly pick n binary features of dataset X_full"""
    categories = np.array(lib.get_categories_full_cat_data(X_full[categorical_columns]))
    categorical_columns = np.array(categorical_columns)
    closest_to_balance = np.argsort(np.abs(X_full.shape[0]//2 - X_full[categorical_columns[categories == 2]].apply(LabelEncoder().fit_transform).sum(axis = 0)).values)
    print(closest_to_balance)
    # multilabel_targets = np.random.choice(categorical_columns[categories == 2], size = n, replace = False)
    multilabel_targets = categorical_columns[categories == 2][closest_to_balance[:n]]
    print(multilabel_targets)
    os.makedirs(multilabel_split_path, exist_ok=True)
    pickle_path = os.path.join(multilabel_split_path, '{}_n{}_multilabel_split.pickle'.format(ds_id, n))

    if os.path.isfile(pickle_path):
    # Load data (deserialize)
        with open(pickle_path, 'rb') as handle:
            multilabel_split_dict = pickle.load(handle)
    else:
        multilabel_split_dict = {'n': n,
                                 'binary_multilabel_targets': multilabel_targets,
                                 'ds_id': ds_id}
        # Store data (serialize)
        with open(pickle_path, 'wb') as handle:
            pickle.dump(multilabel_split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return multilabel_split_dict


def sanity_check_split(X, y, ds_id, p=0.5, split_path=os.path.join('transfer', "sanity_check_splits/")):
    os.makedirs(split_path, exist_ok=True)
    pickle_path = os.path.join(split_path, '{}_p{}_sanity_split.pickle'.format(ds_id, p))

    if os.path.isfile(pickle_path):
        # Load data (deserialize)
        with open(pickle_path, 'rb') as handle:
            split_indices = pickle.load(handle)
    else:
        indices = np.arange(len(X))
        shuffled = np.random.choice(indices, len(indices), replace = False)
        split_indices = {
            'sanity_check_pretrain': shuffled[:int(p * len(shuffled))],
            'sanity_check_downstream': shuffled[int(p * len(shuffled)):],
            'pretrain_proportion': p
        }
        with open(pickle_path, 'wb') as handle:
            pickle.dump(split_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return split_indices


def data_prep_openml_transfer(ds_id, seed, task, stage='pretrain', datasplit=[.65, .15, .2],
                              pretrain_proportion=0.5, downstream_samples_per_class = 2, column_mode=None, pretrain_subsample = False):

    # Arpit - Added column_mode in order to change the data accordingly
    """pretrain proportion is used by multiclass as pretrain_proportion, by regression as quantile and by binary as current experiment number
    downstream_samples_per_class should be 2 5 10 50 250 for multiclass and 10 25 50 250 1250 for regression as data on downstream
    """
    input_seed = seed
    #switch off resampling:
    seed = 0
    if (ds_id == 'mimic') and (pretrain_proportion == 2):
        seed = 1

    #Let's not waste the data on validation set in case the dataset is small
    if 'downstream' in stage:
        datasplit[0] = datasplit[0] + datasplit[1]
        datasplit[1] = 0.0

    np.random.seed(seed)
    # torch.manual_seed(seed)
    # random.seed(seed)
    #If ft-transformer data
    if (type(ds_id) == str) and (not ds_id == 'mimic'):
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
    elif (type(ds_id) == int) and (not ds_id == 'mimic'):
        dataset = openml.datasets.get_dataset(ds_id)
        # load data
        X_full, y_full, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

        # get cat, cont columns
        # categorical_columns, numerical_columns = get_cont_cat_features(ds_id)
        categorical_columns = list(X_full.columns[np.array(categorical_indicator)])
        numerical_columns = list(X_full.columns[~np.array(categorical_indicator)])
    elif ds_id == 'mimic':
        return data_prep_transfer_mimic(ds_id, seed, task, stage, pretrain_proportion,
                                 downstream_samples_per_class, column_mode, pretrain_subsample, input_seed)
        # Arpit - Added column_mode
    else:
        raise ValueError('Wrong ds_id type. Type of ds_id can only be int or str')
    print(numerical_columns)
    print(categorical_columns)
    # print(X_full.shape, y_full.shape, y_full.dtype)
    if (len(y_full.shape) == 1):
        if (y_full.dtype.name == "category"):
            y_full = y_full.apply(str).astype('object')


    if len(X_full) > 1000000:#ds_id in [42728, 42705, 42729, 42571]:
        # import ipdb; ipdb.set_trace()
        #NEED TO RERUN ON LARGER DATASETS!
        sample_idx = X_full.sample(n=1000000, random_state = seed).index
        X_full, y_full = X_full.iloc[sample_idx], y_full.iloc[sample_idx]
        X_full.reset_index(drop=True, inplace=True)
        y_full.reset_index(drop=True, inplace=True)

    #Split into pretrain/downstream datasets
    if task == 'multiclass':
        if stage in ['pretrain', 'downstream']:
            classes = class_split(y_full, ds_id, p=pretrain_proportion)
            split_indices = split_indices_by_class(X_full, y_full, classes[stage])
            X, y = X_full[split_indices], y_full[split_indices]
        elif stage in ['sanity_check_pretrain', 'sanity_check_downstream']:
            print('\n PERFORMING SANITY CHECK \n')
            split_indices = sanity_check_split(X_full, y_full, ds_id, p=pretrain_proportion)[stage]
            X, y = X_full.iloc[split_indices], y_full.iloc[split_indices]
        else:
            raise ValueError('Wrong value for stage')

    elif task == 'regression':
        if stage in ['pretrain', 'downstream']:
            #pretrain proportion argument acts as quantile in regression setting
            regression_split_dict = regression_split(y_full, ds_id, q=pretrain_proportion, direction = 'downstream_ge_q')
            split_indices = regression_split_indices(X_full, y_full, regression_split_dict, stage)
            X, y = X_full[split_indices], y_full[split_indices]
    else:
        raise NotImplementedError('Transfer learning data strategies are implemented only for multiclass, regression or binclass so far')

    for col in categorical_columns:
        X[col] = X[col].apply(str).astype("object")

    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    X[categorical_columns] = X[categorical_columns].fillna("MissingValue")

    # split data into train/val/test
    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))
    train_indices = X[X.Set=="train"].index
    replacement_sampling = False
    if 'downstream' in stage:
        #switching to downstream_samples_per_class
        if (downstream_samples_per_class*(len(set(y)) if task == 'multiclass' else 5) <= len(X[X.Set=="train"])):# or (ds_id in ['aloi']): #aloi is huge 1000 classes, don't wanna oversample
            #sample without replacement
            if (task == 'multiclass') or (task == 'binclass'):
                train_X, _, = train_test_split(X[X.Set == 'train'],
                                               train_size=downstream_samples_per_class * len(set(y)),
                                               stratify=y[X.Set == 'train'], random_state = seed)
                train_indices = train_X.index
                # train_indices = X[X.Set=="train"].sample(n=downstream_samples_per_class*len(set(y)), random_state = seed).index
            elif task == 'regression':
                train_indices = X[X.Set == "train"].sample(n=downstream_samples_per_class * 5, random_state=seed).index
            else:
                raise ValueError('Task should be multiclass, binclass or regression. You are using {}'.format(task))
            replacement_sampling = False
        else:
            raise NotImplementedError('Sampling with replacement is probably not helpful')
            #sample with replacement
            # train_indices = X[X.Set == "train"].sample(n=downstream_samples_per_class * len(set(y)),
            #                                            random_state=seed, replace = True).index
            # replacement_sampling = True
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    if (task != 'regression') and (task != 'binclass'):
        l_enc = LabelEncoder()
        if not (len(y.shape) > 1):
            y = l_enc.fit_transform(y)
        else:
            y = y.apply(l_enc.fit_transform)
    else:
        y = y.to_numpy()

    X_cat_train = X[categorical_columns].values[train_indices]
    X_num_train = X[numerical_columns].values[train_indices]
    # if ('pretrain' in stage) and (task == 'binclass'):
    #     y_train = y.values[train_indices]
    # else:
    y_train = y[train_indices]

    X_cat_val = X[categorical_columns].values[valid_indices]
    X_num_val = X[numerical_columns].values[valid_indices]
    # if ('pretrain' in stage) and (task == 'binclass'):
    #     y_val = y.values[valid_indices]
    # else:
    y_val = y[valid_indices]

    X_cat_test = X[categorical_columns].values[test_indices]
    X_num_test = X[numerical_columns].values[test_indices]
    # if ('pretrain' in stage) and (task == 'binclass'):
    #     y_test = y.values[test_indices]
    # else:
    y_test = y[test_indices]


    info = {}
    info['name'] = ds_id
    info['stage'] = stage
    info['split'] = seed
    # info['split'] = datasplit
    info['task_type'] = task
    info['n_num_features'] = len(numerical_columns)
    info['n_cat_features'] = len(categorical_columns)
    info['train_size'] = len(train_indices)
    info['val_size'] = len(valid_indices)
    info['test_size'] = len(test_indices)
    info['replacement_sampling'] = replacement_sampling
    if task == 'multiclass':
        info['n_classes'] = len(set(y))
    if task == 'binclass':
        if len(y_train.shape) > 1:
            info['n_classes'] = y_train.shape[1]
        else:
            info['n_classes'] = 1

    if len(numerical_columns) > 0:
        #We have a zero size validation set, replace it with train (hack)
        if 'downstream' in stage:
            X_num_val = X_num_train
        N = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test}
    else:
        N = None

    if len(categorical_columns) > 0:
        # We have a zero size validation set, replace it with train (hack)
        if 'downstream' in stage:
            X_cat_val = X_cat_train
        C = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test}
    else:
        C = None
    # N = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test}
    # C = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test}
    # We have a zero size validation set, replace it with train (hack)
    if 'downstream' in stage:
        y_val = y_train
    y = {'train': y_train, 'val': y_val, 'test': y_test}

    if len(categorical_columns) > 0:
        full_cat_data_for_encoder = X_full[categorical_columns]
    else:
        full_cat_data_for_encoder = None
    return N, C, y, info, full_cat_data_for_encoder


def data_prep_transfer_mimic(ds_id, seed, task, stage='pretrain', pretrain_proportion=0, downstream_samples_per_class = 2, column_mode=None, pretrain_subsample = False, input_seed=0):
    """pretrain proportion is used by multiclass as pretrain_proportion, by regression as quantile and by binary as current experiment number
    downstream_samples_per_class should be 2 5 10 50 250 for multiclass and 10 25 50 250 1250 for regression as data on downstream
    """
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # random.seed(seed)
    #If ft-transformer data
    mimic_target_columns = ['diabetes_diagnosed', 'hypertensive_diagnosed', 'ischematic_diagnosed',
                            'heart_diagnosed', 'overweight_diagnosed', 'anemia_diagnosed', 'respiratory_diagnosed',
                            'hypotension_diagnosed', 'lipoid_diagnosed', 'atrial_diagnosed', 'purpura_diagnosed',
                            'alcohol_diagnosed']
    X_train = pd.read_csv('data/mimic_train_X.csv')
    X_val = pd.read_csv('data/mimic_val_X.csv')
    X_test = pd.read_csv('data/mimic_test_X.csv')
    y_train_full = pd.read_csv('data/mimic_train_y.csv')
    y_val_full = pd.read_csv('data/mimic_val_y.csv')
    y_test_full = pd.read_csv('data/mimic_test_y.csv')
    categorical_columns = ['gender']
    numerical_columns = list(X_train.columns[X_train.columns != 'gender'])
    X_train[categorical_columns] = X_train[categorical_columns].fillna("MissingValue")
    X_val[categorical_columns] = X_val[categorical_columns].fillna("MissingValue")
    X_test[categorical_columns] = X_test[categorical_columns].fillna("MissingValue")
    print(numerical_columns)
    print(categorical_columns)

    # Arpit -  even though we will have regression, we still need the following data

    if 'downstream' in stage:
        #Merge validation set into train, keep the dummy validation set for the code not to fail
        y_train_full = pd.concat([y_train_full, y_val_full], ignore_index=True)
        X_train = pd.concat([X_train, X_val], ignore_index=True)
        print('Using downstream target:', mimic_target_columns[pretrain_proportion])
        y_train = y_train_full[mimic_target_columns[pretrain_proportion]]
        y_val = y_val_full[mimic_target_columns[pretrain_proportion]]
        y_test = y_test_full[mimic_target_columns[pretrain_proportion]]
    elif 'pretrain' in stage:
        #Do multitarget in regular pretrain
        print('Dropping downstream target:', mimic_target_columns[pretrain_proportion])
        y_train = y_train_full.drop(columns=[mimic_target_columns[pretrain_proportion]])
        y_val = y_val_full.drop(columns=[mimic_target_columns[pretrain_proportion]])
        y_test = y_test_full.drop(columns=[mimic_target_columns[pretrain_proportion]])
        if pretrain_subsample:
            subsample_tuning_target = np.random.randint(10)
            possible_tuning_targets = np.array(y_train.columns)
            print('Using subsample tuning target:', possible_tuning_targets[subsample_tuning_target])
            y_train = y_train_full[possible_tuning_targets[subsample_tuning_target]]
            y_val = y_val_full[possible_tuning_targets[subsample_tuning_target]]
            y_test = y_test_full[possible_tuning_targets[subsample_tuning_target]]
    else:
        raise ValueError('Stage is incorrect!')

    # Arpit - handling the drop of the column
    to_drop_index = 0  # the index of the categorical column we are removing
    to_drop = numerical_columns[to_drop_index]

    if column_mode == 'remove_column':
        X_train = X_train.drop(columns=[to_drop])
        X_val = X_val.drop(columns=[to_drop])
        X_test = X_test.drop(columns=[to_drop])

        print("All Cols")
        print(numerical_columns)
        print("To drop")
        print(to_drop)

        numerical_columns.remove(to_drop)

    # Arpit - Also since we are dropping the numerical column, just train to predict a value.
    # Arpit - Before I made a lot of changes to handle categorical coloumn being dropped.

    elif column_mode == 'predict_missing_column' or column_mode == 'train_to_predict_missing_column':
        y_train = X_train[to_drop]
        y_val = X_val[to_drop]
        y_test = X_test[to_drop]

        X_train = X_train.drop(columns=[to_drop])
        X_val = X_val.drop(columns=[to_drop])
        X_test = X_test.drop(columns=[to_drop])

        print("All Cols")
        print(numerical_columns)
        print("To drop")
        print(to_drop)
        numerical_columns.remove(to_drop)


    X_train_full = X_train.copy()
    y_train_full = y_train.copy()
    if ('downstream' in stage) or pretrain_subsample:
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

    if column_mode == 'train_to_predict_missing_column':
        # drop the nas
        Na_indexes = y_train.copy().isna()
        y_train = y_train[~Na_indexes]
        X_train = X_train[~Na_indexes]

        Na_indexes = y_val.copy().isna()
        y_val = y_val[~Na_indexes]
        X_val = X_val[~Na_indexes]

        Na_indexes = y_test.copy().isna()
        y_test = y_test[~Na_indexes]
        X_test = X_test[~Na_indexes]


    X_cat_train = X_train[categorical_columns].values
    X_num_train = X_train[numerical_columns].values
    y_train = y_train.values.astype('float')

    X_cat_val = X_val[categorical_columns].values
    X_num_val = X_val[numerical_columns].values
    y_val = y_val.values.astype('float')

    X_cat_test = X_test[categorical_columns].values
    X_num_test = X_test[numerical_columns].values
    y_test = y_test.values.astype('float')

    # Arpit -
    # when training on the missing upstream feature, we have trained on the downstream task for different samples
    # but when there is a missing upstream feature, we have all the data and nothing for downstream samples
    if column_mode == 'train_with_imputed_column':
        if 'downstream' in stage:
            overwrite = torch.load(
                f"./predict_missing_column/predicted_column_using_upstream_on_downstream_train_mimic_{input_seed}_{downstream_samples_per_class}_{pretrain_proportion}.pt").int().cpu().numpy()

            print("Sanity_rmse")
            print(np.sqrt(np.mean( (X_num_train[:, to_drop_index] - overwrite)**2)) )

            X_num_train[:, to_drop_index] = overwrite

            overwrite = torch.load(
                f"./predict_missing_column/predicted_column_using_upstream_on_downstream_val_mimic_{input_seed}_{downstream_samples_per_class}_{pretrain_proportion}.pt").int().cpu().numpy()
            X_num_val[:, to_drop_index] = overwrite

            overwrite = torch.load(
                f"./predict_missing_column/predicted_column_using_upstream_on_downstream_test_mimic_{input_seed}_{downstream_samples_per_class}_{pretrain_proportion}.pt").int().cpu().numpy()
            X_num_test[:, to_drop_index] = overwrite

        elif 'pretrain' in stage:
            overwrite = torch.load(
                f"./predict_missing_column/predicted_column_using_downstream_on_upstream_train_mimic_{input_seed}_{downstream_samples_per_class}_{pretrain_proportion}.pt").int().cpu().numpy()

            print("Sanity_rmse")
            print(np.sqrt(np.mean( (X_num_train[:, to_drop_index] - overwrite)**2)) )

            X_num_train[:, to_drop_index] = overwrite

            overwrite = torch.load(
                f"./predict_missing_column/predicted_column_using_downstream_on_upstream_val_mimic_{input_seed}_{downstream_samples_per_class}_{pretrain_proportion}.pt").int().cpu().numpy()
            X_num_val[:, to_drop_index] = overwrite

            overwrite = torch.load(
                f"./predict_missing_column/predicted_column_using_downstream_on_upstream_test_mimic_{input_seed}_{downstream_samples_per_class}_{pretrain_proportion}.pt").int().cpu().numpy()
            X_num_test[:, to_drop_index] = overwrite

    info = {}
    info['name'] = ds_id
    info['stage'] = stage
    info['split'] = seed
    # info['split'] = datasplit
    info['task_type'] = task
    info['n_num_features'] = len(numerical_columns)
    info['n_cat_features'] = len(categorical_columns)
    info['train_size'] = X_train.shape[0]
    info['val_size'] = X_val.shape[0]
    info['test_size'] = X_test.shape[0]
    info['replacement_sampling'] = False

    # Arpit - Ask if this will be a problem in regression

    if task == 'multiclass':
        info['n_classes'] = len(set(y))
    if task == 'binclass':
        if len(y_train.shape) > 1:
            info['n_classes'] = y_train.shape[1]
        else:
            info['n_classes'] = 1

    if len(numerical_columns) > 0:
        #We have a zero size validation set, replace it with train (hack)
        if 'downstream' in stage:
            X_num_val = X_num_train
        N = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test}
    else:
        N = None

    if len(categorical_columns) > 0:
        # We have a zero size validation set, replace it with train (hack)
        if 'downstream' in stage:
            X_cat_val = X_cat_train
        C = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test}
    else:
        C = None

    # We have a zero size validation set, replace it with train (hack)
    if 'downstream' in stage:
        y_val = y_train
    y = {'train': y_train, 'val': y_val, 'test': y_test}
    print('\n Train size:{} Val size:{} Test size:{}'.format(len(y_train), len(y_val), len(y_test)))
    if len(categorical_columns) > 0:
        #this only works with mimic since the only categorical feature is gender
        full_cat_data_for_encoder = X_train_full[categorical_columns]
    else:
        full_cat_data_for_encoder = None
    return N, C, y, info, full_cat_data_for_encoder

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