import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
import pandas as pd

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

def split_indices_by_class(X, y, keep_classes):
    return y.isin(keep_classes)

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
                              pretrain_proportion=0.5, downstream_train_data_limit = 1.0):

    np.random.seed(seed)
    dataset = openml.datasets.get_dataset(ds_id)
    # load data
    X_full, y_full, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

    # get cat, cont columns
    # categorical_columns, numerical_columns = get_cont_cat_features(ds_id)
    # categorical_columns = X_full.columns[list(np.where(np.array(categorical_indicator) == True)[0])].tolist()
    # numerical_columns = list(set(X_full.columns.tolist()) - set(categorical_columns))
    categorical_columns = list(X_full.columns[np.array(categorical_indicator)])
    numerical_columns = list(X_full.columns[~np.array(categorical_indicator)])
    print(numerical_columns)
    print(categorical_columns)
    print(X_full.shape, y_full.shape, y_full.dtype)
    if y_full.dtype == "category":
        y_full = y_full.apply(str).astype('object')

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
        #So far, we only handle dataset 126
        assert ds_id == 126
        X_source, y_source, X_target, y_target, cont_columns, cat_columns = split_126(X_full, y_full, numerical_columns, categorical_columns)
        if 'pretrain' in stage:
            X, y = X_source, y_source
        elif 'downstream' in stage:
            X, y = X_target, y_target
        else:
            raise ValueError('Only pretrain or downstream stages are allowed')
    else:
        raise NotImplementedError('Transfer learning data strategies are implemented only for multiclass or binclass so far')

    for col in categorical_columns:
        X[col] = X[col].apply(str).astype("object")

    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    X[categorical_columns] = X[categorical_columns].fillna("MissingValue")

    # split data into train/val/test
    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))
    train_indices = X[X.Set=="train"].index
    if 'downstream' in stage:
        train_indices = X[X.Set=="train"].sample(frac=downstream_train_data_limit).index#[:int(downstream_train_data_limit * len(train_indices))]
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