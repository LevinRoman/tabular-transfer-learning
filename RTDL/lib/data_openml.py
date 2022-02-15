import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
import pandas as pd
import lib
import torch

def get_cont_cat_features(dataset_id):
    if dataset_id == 6:
        #this one does not work for some reason
        continuous_cols = []
        cat_embed_cols = ['x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar',
                           'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy',
                           'y-ege', 'yegvx']
    if dataset_id == 261:
        cat_embed_cols = []
        continuous_cols = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6',
                           'input7', 'input8', 'input9', 'input10', 'input11', 'input12',
                           'input13', 'input14', 'input15', 'input16']
    if dataset_id == 42078:
        cat_embed_cols = []
        continuous_cols = ['brewery_id', '']
    if dataset_id == 1381:
        cat_embed_cols = []
        continuous_cols = ['x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar',
                          'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br',
                          'x-ege', 'xegvy', 'y-ege', 'yegvx']
    if dataset_id == 1113:
        cat_embed_cols = ['protocol_type', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']
        #'service',
        continuous_cols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent',
       'hot', 'num_failed_logins', 'lnum_compromised', 'lroot_shell',
       'lsu_attempted', 'lnum_root', 'lnum_file_creations', 'lnum_shells',
       'lnum_access_files', 'lnum_outbound_cmds', 'count', 'srv_count',
       'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
       'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
       'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate']
    if dataset_id == 1503:
        cat_embed_cols = ['V14']
        continuous_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13']
    if dataset_id == 54:
        cat_embed_cols = ["SKEWNESS_ABOUT_MINOR"]
        continuous_cols = ["COMPACTNESS", "CIRCULARITY", "DISTANCE_CIRCULARITY", "RADIUS_RATIO", "PR.AXIS_ASPECT_RATIO",
                           "MAX.LENGTH_ASPECT_RATIO", "SCATTER_RATIO", "ELONGATEDNESS", "PR.AXIS_RECTANGULARITY", "MAX.LENGTH_RECTANGULARITY",
                           "SCALED_VARIANCE_MAJOR", "SCALED_VARIANCE_MINOR", "SCALED_RADIUS_OF_GYRATION", "SKEWNESS_ABOUT_MAJOR",
                           "KURTOSIS_ABOUT_MAJOR", "KURTOSIS_ABOUT_MINOR", "HOLLOWS_RATIO"]

    if dataset_id == 42164:
        cat_embed_cols =['body_type', 'diet', 'drinks', 'drugs', 'education', 'essay0', 'essay1', 'essay2', 'essay3', 'essay4',
                         'essay5', 'essay6', 'essay7', 'essay8', 'essay9', 'ethnicity', 'job', 'last_online',
                         'location', 'offspring', 'orientation', 'pets', 'religion', 'sex', 'sign', 'smokes', 'speaks', 'status']
        continuous_cols = ['height', 'income']
    if dataset_id == 42998:
        cat_embed_cols = ['holiday', 'weather_main', 'weather_description', 'date_time']
        continuous_cols = ['temp', 'rain_1h', 'snow_1h', 'clouds_all']
    if dataset_id == 40536:
        cat_embed_cols = ['has_null', 'gender', 'd_d_age', 'race', 'race_o', 'samerace', 'd_importance_same_race', 'd_importance_same_religion',
                          'field','d_pref_o_attractive', 'd_pref_o_sincere', 'd_pref_o_intelligence', 'd_pref_o_funny', 'd_pref_o_ambitious', 'd_pref_o_shared_interests',
                          'd_attractive_o', 'd_sinsere_o', 'd_intelligence_o', 'd_funny_o', 'd_ambitous_o', 'd_shared_interests_o',
                          'd_attractive_important', 'd_sincere_important', 'd_intellicence_important', 'd_funny_important', 'd_ambtition_important', 'd_shared_interests_important',
                          'd_attractive', 'd_sincere', 'd_intelligence', 'd_funny', 'd_ambition',
                          'd_attractive_partner', 'd_sincere_partner', 'd_intelligence_partner', 'd_funny_partner', 'd_ambition_partner', 'd_shared_interests_partner',
                          'd_sports', 'd_tvsports', 'd_exercise', 'd_dining', 'd_museums', 'd_art', 'd_hiking', 'd_gaming', 'd_clubbing', 'd_reading', 'd_tv', 'd_theater', 'd_movies', 'd_concerts', 'd_music', 'd_shopping', 'd_yoga',
                          'd_interests_correlate',  'd_expected_happy_with_sd_people', 'd_expected_num_interested_in_me', 'd_expected_num_matches',
                          'd_like', 'd_guess_prob_liked']
        continuous_cols = ['wave', 'age', 'age_o', 'd_age', 'importance_same_race', 'importance_same_religion',
                           'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence', 'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests',
                           'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o', 'shared_interests_o',
                           'attractive_important', 'sincere_important', 'intellicence_important', 'funny_important', 'ambtition_important', 'shared_interests_important',
                           'attractive', 'sincere', 'intelligence', 'funny', 'ambition',
                           'attractive_partner', 'sincere_partner', 'intelligence_partner', 'funny_partner', 'ambition_partner', 'shared_interests_partner',
                           'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga',
                           'interests_correlate', 'expected_happy_with_sd_people', 'expected_num_interested_in_me', 'expected_num_matches',
                           'like', 'guess_prob_liked', 'met']

    if dataset_id == 1590:
        cat_embed_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
        continuous_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    if dataset_id == 1502:
        cat_embed_cols = []
        continuous_cols = ['V1', 'V2', 'V3']
    if dataset_id == 139:
        cat_embed_cols = ['Number_of_cars', 'Number_of_different_loads', 'num_wheels_2', 'length_2', 'shape_2', 'num_loads_2', 'load_shape_2',
                          'num_wheels_3', 'length_3', 'shape_3', 'num_loads_3', 'load_shape_3', 'num_wheels_4', 'length_4', 'shape_4', 'num_loads_4',
                          'load_shape_4', 'num_wheels_5', 'length_5', 'shape_5', 'num_loads_5', 'load_shape_5', 'Rectangle_next_to_rectangle',
                          'Rectangle_next_to_triangle', 'Rectangle_next_to_hexagon', 'Rectangle_next_to_circle', 'Triangle_next_to_triangle',
                          'Triangle_next_to_hexagon', 'Triangle_next_to_circle', 'Hexagon_next_to_hexagon', 'Hexagon_next_to_circle', 'Circle_next_to_circle']
        continuous_cols = []
    if dataset_id == 162:
        cat_embed_cols = []
        continuous_cols = ['attrib1', 'attrib2', 'attrib3']
    if dataset_id == 126:
        cat_embed_cols = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'installment_commitment',
                          'personal_status', 'other_parties', 'residence_since', 'property_magnitude', 'age', 'other_payment_plans', 'housing', 'existing_credits',
                          'job', 'num_dependents', 'own_telephone', 'foreign_worker']
        continuous_cols = []
    if dataset_id == 143:
        cat_embed_cols = ['handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
                          'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback',
                          'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
        continuous_cols = []
    if dataset_id == 257:
        cat_embed_cols = ['surgery', 'Age', 'temp_extremities', 'peripheral_pulse', 'mucous_membranes',
                          'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distension', 'nasogastric_tube', 'nasogastric_reflux',
                          'rectal_examination', 'abdomen', 'abdominocentesis_appearance', 'outcome']
        continuous_cols = ['rectal_temperature', 'pulse', 'respiratory_rate', 'nasogastric_reflux_PH',  'packed_cell_volume', 'total_protein', 'abdomcentesis_total_protein']

    if dataset_id == 1483:
        # cat_embed_cols = ['V1', 'V2']
        cat_embed_cols = []
        continuous_cols = ['V3', 'V4', 'V5', 'V6', 'V7']

    return cat_embed_cols, continuous_cols




def data_prep_openml(ds_id, seed, task, datasplit=[.65, .15, .2]):

    np.random.seed(seed)
    dataset = openml.datasets.get_dataset(ds_id)
    # load data
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    # get cat, cont columns
    categorical_columns, numerical_columns = get_cont_cat_features(ds_id)


    for col in categorical_columns:
        X[col] = X[col].apply(str).astype("object")

    X[categorical_columns] = X[categorical_columns].fillna("MissingValue")

    # split data into train/val/test
    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))
    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    if task != 'regression':
        l_enc = LabelEncoder()
        y = l_enc.fit_transform(y)
    else:
        y = y.to_numpy()

    X_cat_train = X[categorical_columns].values[train_indices]
    X_num_train = X[numerical_columns].values[train_indices]
    y_train = y[train_indices]

    X_cat_val = X[categorical_columns].values[valid_indices]
    X_num_val = X[numerical_columns].values[valid_indices]
    y_val = y[valid_indices]

    X_cat_test = X[categorical_columns].values[test_indices]
    X_num_test = X[numerical_columns].values[test_indices]
    y_test = y[test_indices]


    info = {}
    info['name'] = ds_id
    info['split'] = seed
    info['task_type'] = task
    info['n_num_features'] = len(numerical_columns)
    info['n_cat_features'] = len(categorical_columns)
    info['train_size'] = len(train_indices)
    info['val_size'] = len(valid_indices)
    info['test_size'] = len(test_indices)
    if task == 'multiclass':
        info['n_classes'] = len(set(y))

    if len(numerical_columns) > 0:
        N = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test}
    else:
        N = None

    if len(categorical_columns) > 0:
        C = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test}
    else:
        C = None
    # N = {'train': X_num_train, 'val': X_num_val, 'test': X_num_test}
    # C = {'train': X_cat_train, 'val': X_cat_val, 'test': X_cat_test}
    y = {'train': y_train, 'val': y_val, 'test': y_test}

    return N, C, y, info

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
                              pretrain_proportion=0.5, downstream_samples_per_class = 2):
    """pretrain proportion is used by multiclass as pretrain_proportion, by regression as quantile and by binary as current experiment number
    downstream_samples_per_class should be 2 5 10 50 250 for multiclass and 10 25 50 250 1250 for regression as data on downstream
    """
    #Let's not waste the data on validation set in case the dataset is small
    if 'downstream' in stage:
        datasplit[0] = datasplit[0] + datasplit[1]
        datasplit[1] = 0.0

    np.random.seed(seed)
    # torch.manual_seed(seed)
    # random.seed(seed)
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

    if len(X_full) > 100000:#ds_id in [42728, 42705, 42729, 42571]:
        # import ipdb; ipdb.set_trace()
        #NEED TO RERUN ON LARGER DATASETS!
        sample_idx = X_full.sample(n=100000, random_state = seed).index
        X_full, y_full = X_full.iloc[sample_idx], y_full.iloc[sample_idx]
        X_full.reset_index(drop=True, inplace=True)
        y_full.reset_index(drop=True, inplace=True)

    if (task == 'binclass') and ('pretrain' in stage):
        if len(X_full) > 100000:  # ds_id in [42728, 42705, 42729, 42571]:
            # import ipdb; ipdb.set_trace()
            sample_idx = X_full.sample(n=100000, random_state = seed).index
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
    elif task == 'binclass':
        if len(categorical_columns) > 0:
            #append target as one new feature:
            X_full = pd.concat([X_full, y_full], axis = 1)
            #if the target is categorical, add it to categorical columns
            if ds_id in [1111, 1017, 1596, 4541, 40664]:
                categorical_columns.append(y_full.name)

            multilabel_targets = binary_multilabel_targets(ds_id,
                                                                  X_full,
                                                                  categorical_columns,
                                                                  n=5)['binary_multilabel_targets']

            for col in categorical_columns:
                X_full[col] = X_full[col].apply(str).astype("object")

            y_full = X_full[multilabel_targets]
            X_full.drop(columns = multilabel_targets, inplace = True)
            #remove target names from categorical column names
            categorical_columns = [col for col in categorical_columns if col not in multilabel_targets]

            X = X_full.copy()

            if 'downstream' in stage:
                y = y_full[multilabel_targets[pretrain_proportion]]
            elif 'pretrain' in stage:
                y = y_full.drop(columns = [multilabel_targets[pretrain_proportion]])
        else:
            raise ValueError('Datasets for binary transfer task cannot have 0 categorical features!')
        #Old experiment, domain shift, we only handle dataset 126
        # assert ds_id == 126
        # X_source, y_source, X_target, y_target, cont_columns, cat_columns = split_126(X_full, y_full, numerical_columns, categorical_columns)
        # if 'pretrain' in stage:
        #     X, y = X_source, y_source
        # elif 'downstream' in stage:
        #     X, y = X_target, y_target
        # else:
        #     raise ValueError('Only pretrain or downstream stages are allowed')
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
                train_indices = X[X.Set=="train"].sample(n=downstream_samples_per_class*len(set(y)), random_state = seed).index
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
    if task != 'regression':
        l_enc = LabelEncoder()
        if not (len(y.shape) > 1):
            y = l_enc.fit_transform(y)
        else:
            y = y.apply(l_enc.fit_transform)
    else:
        y = y.to_numpy()

    X_cat_train = X[categorical_columns].values[train_indices]
    X_num_train = X[numerical_columns].values[train_indices]
    if ('pretrain' in stage) and (task == 'binclass'):
        y_train = y.values[train_indices]
    else:
        y_train = y[train_indices]

    X_cat_val = X[categorical_columns].values[valid_indices]
    X_num_val = X[numerical_columns].values[valid_indices]
    if ('pretrain' in stage) and (task == 'binclass'):
        y_val = y.values[valid_indices]
    else:
        y_val = y[valid_indices]

    X_cat_test = X[categorical_columns].values[test_indices]
    X_num_test = X[numerical_columns].values[test_indices]
    if ('pretrain' in stage) and (task == 'binclass'):
        y_test = y.values[test_indices]
    else:
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

def split_126(X, y, cont_columns, cat_columns, order = 1):
    if order == 1:
        X_target = X[(X['credit_history'] == 'critical/other existing credit')].reset_index(drop = True)
        X_source = X[(X['credit_history'] == 'all paid') ].reset_index(drop = True)
        y_target = y[(X['credit_history'] == 'critical/other existing credit')].reset_index(drop = True)
        y_source = y[(X['credit_history'] == 'all paid') ].reset_index(drop = True)

    else:
        X_source = X[(X['credit_history'] == 'critical/other existing credit') ].reset_index(drop = True)
        X_target = X[(X['credit_history'] == 'all paid') ].reset_index(drop = True)
        y_source = y[(X['credit_history'] == 'critical/other existing credit') ].reset_index(drop = True)
        y_target = y[(X['credit_history'] == 'all paid') ].reset_index(drop = True)

    print('Source label distribution: ', y_source.value_counts())
    print('Target label distribution: ', y_target.value_counts())

    source_pts_neg = np.random.choice(X_source[y_source == 'bad'].index, 20000, replace = False)
    source_pts_pos = np.random.choice(X_source[y_source == 'good'].index, 20000, replace = False)
    target_pts_neg = np.random.choice(X_target[y_target == 'bad'].index, 20000, replace = False)
    target_pts_pos = np.random.choice(X_target[y_target == 'good'].index, 20000, replace = False)

    X_source = X_source.iloc[list(source_pts_neg) + list(source_pts_pos)].reset_index(drop=True)
    y_source = y_source.iloc[list(source_pts_neg) + list(source_pts_pos)].reset_index(drop=True)
    X_target = X_target.iloc[list(target_pts_neg) + list(target_pts_pos)].reset_index(drop=True)
    y_target = y_target.iloc[list(target_pts_neg) + list(target_pts_pos)].reset_index(drop=True)

    print('Source label distribution: ', y_source.value_counts())
    print('Target label distribution: ', y_target.value_counts())


    '''Dropping features and getting indices for cont/cat features'''
    X_source = X_source.drop(['credit_history'], axis=1)
    X_target = X_target.drop(['credit_history'], axis=1)
    cat_columns.remove('credit_history')

    return X_source, y_source, X_target, y_target, cont_columns, cat_columns