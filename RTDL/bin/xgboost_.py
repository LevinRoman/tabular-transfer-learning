from copy import deepcopy
from pathlib import Path
import os
import numpy as np
import zero
from xgboost import XGBClassifier, XGBRegressor

import lib

args, output = lib.load_config()
args['model']['random_state'] = args['seed']

zero.set_randomness(args['seed'])
dset_id = args['data']['dset_id']
stats = lib.load_json(output / 'stats.json')
stats.update({'dataset': dset_id, 'algorithm': Path(__file__).stem})

# Prepare data and model
#####################################################################################
# TRANSFER#
#####################################################################################
N, C, y, info, full_cat_data_for_encoder = lib.data_prep_openml_transfer(ds_id=dset_id,
                                                                         seed=args['seed'],
                                                                         task=args['data']['task'],
                                                                         stage=args['transfer']['stage'],
                                                                         datasplit=[.65, .15, .2],
                                                                         pretrain_proportion=args['transfer'][
                                                                             'pretrain_proportion'],
                                                                         downstream_samples_per_class=args['transfer'][
                                                                             'downstream_samples_per_class'])
#####################################################################################
# TRANSFER#
#####################################################################################

stats['replacement_sampling'] = info['replacement_sampling']
if args['data']['task'] == 'multiclass':
    stats['num_classes_train'] = len(set(y['train']))
    stats['num_classes_test'] = len(set(y['test']))
else:
    stats['num_classes_train'] = np.nan
    stats['num_classes_test'] = np.nan

stats['num_training_samples'] = len(y['train'])
if C is not None:
    stats['cat_features_no'] = C['train'].shape[1]
else:
    stats['cat_features_no'] = 0
if N is not None:
    stats['num_features_no'] = N['train'].shape[1]
else:
    stats['num_features_no'] = 0

if len(set(y['train'])) == 2:
    info['task_type'] = 'binclass'

D = lib.Dataset(N, C, y, info)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'ohe'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
    full_cat_data_for_encoder = full_cat_data_for_encoder
)
if C is None:
    X = X[0]

assert isinstance(X, dict)
zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')

fit_kwargs = deepcopy(args["fit"])
# XGBoost does not automatically use the best model, so early stopping must be used
if not 'downstream' in args['transfer']['stage']:
    assert 'early_stopping_rounds' in fit_kwargs
    fit_kwargs['eval_set'] = [(X[lib.VAL], Y[lib.VAL])]


if D.is_regression:
    model = XGBRegressor(**args["model"])
    predict = model.predict
else:
    # if not 'downstream' in args['transfer']['stage']:
    model = XGBClassifier(**args["model"], disable_default_eval_metric=True)
    # else:
    #     model = XGBClassifier(**args["model"])
    if D.is_multiclass:
        predict = model.predict_proba
        fit_kwargs['eval_metric'] = 'merror'
    else:
        predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
        fit_kwargs['eval_metric'] = 'error'


# Fit model
timer = zero.Timer()
timer.run()

model.fit(X[lib.TRAIN], Y[lib.TRAIN].astype('int'), **fit_kwargs)

# Save model and metrics

model.save_model(str(output / "model.xgbm"))
np.save(output / "feature_importances.npy", model.feature_importances_)

stats['metrics'] = {}
for part in X:
    p = predict(X[part])
    stats['metrics'][part] = lib.calculate_metrics(
        D.info['task_type'], Y[part], p, 'probs', y_info
    )
    np.save(output / f'p_{part}.npy', p)
stats['time'] = lib.format_seconds(timer())
lib.dump_stats(stats, output, True)
lib.backup_output(output)

# if 'downstream' in args['transfer']['stage']:
#     os.remove(output / 'checkpoint.pt')
