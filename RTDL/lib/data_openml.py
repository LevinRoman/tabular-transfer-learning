import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_cont_cat_features(dataset_id):
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
