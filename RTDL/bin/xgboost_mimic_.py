from copy import deepcopy
from pathlib import Path
import os
import numpy as np
import pandas as pd
import zero
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputClassifier

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
                                                                             'downstream_samples_per_class'],
                                                                         pretrain_subsample = args['transfer']['pretrain_subsample'])
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

# if len(set(y['train'])) == 2:
#     info['task_type'] = 'binclass'

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
if ('pretrain' in args['transfer']['stage']) and (not args['transfer']['pretrain_subsample']):
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
        if 'pretrain' in args['transfer']['stage']:
            if not args['transfer']['pretrain_subsample']:
                predictions_train = np.zeros(Y[lib.TRAIN].shape)
                predictions_val = np.zeros(Y[lib.VAL].shape)
                predictions_test = np.zeros(Y[lib.TEST].shape)
                for cur_target in range(Y[lib.TRAIN].shape[1]):
                    model = XGBClassifier(**args["model"], disable_default_eval_metric=True)
                    fit_kwargs['eval_metric'] = 'auc'
                    fit_kwargs['eval_set'] = [(X[lib.VAL], Y[lib.VAL][:, cur_target])]
                    model.fit(X[lib.TRAIN], Y[lib.TRAIN][:, cur_target].astype('int'), **fit_kwargs)
                    model.save_model(str(output / "model_tg{}.xgbm".format(cur_target)))
                    predict = lambda x: model.predict_proba(x)[:, 1]
                    predictions_train[:, cur_target] = predict(X[lib.TRAIN])
                    predictions_val[:, cur_target] = predict(X[lib.VAL])
                    predictions_test[:, cur_target] = predict(X[lib.TEST])
            else:
                model = XGBClassifier(**args["model"], disable_default_eval_metric=True)
                fit_kwargs['eval_metric'] = 'auc'
                predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
        elif 'downstream' in args['transfer']['stage']:
            model = XGBClassifier(**args["model"], disable_default_eval_metric=True)
            fit_kwargs['eval_metric'] = 'auc'
            predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
        else:
            raise ValueError('stage can be only downstream or pretrain')


# Fit model
timer = zero.Timer()
timer.run()
if 'downstream' in args['transfer']['stage']:
    # loading checkpoint of the upstream model, do stacking
    if (args['transfer']['load_checkpoint']):
        print('Stacking')
        for part in X:
            upstream_predictions = []
            for cur_tg in range(11): #loop over 11 upstream models and concat their predictions
                from_file = XGBClassifier(**args["model"], disable_default_eval_metric=True)
                from_file.load_model(args['transfer']['checkpoint_path']+'/model_tg{}.xgbm'.format(cur_tg))
                upstream_predictions.append(from_file.predict(X[part]))
            X[part] = np.concatenate([X[part], np.array(upstream_predictions).T], axis = 1) #note that cat features are specified by index
    else:
        print('No stacking')

if (not D.is_regression) and (not D.is_multiclass) and ('pretrain' in args['transfer']['stage']) and (not args['transfer']['pretrain_subsample']):
    stats['metrics'] = {}
    for part in X:
        if part == lib.TRAIN:
            p = predictions_train
        elif part == lib.VAL:
            p = predictions_val
        elif part == lib.TEST:
            p = predictions_test
        else:
            raise ValueError('Wrong part value!')
        stats['metrics'][part] = lib.calculate_metrics(
            D.info['task_type'], Y[part], p, 'probs', y_info
        )
        # np.save(output / f'p_{part}.npy', p)
else:
    model.fit(X[lib.TRAIN], Y[lib.TRAIN].astype('int'), **fit_kwargs)
    # Save model and metrics
    model.save_model(str(output / "model.xgbm"))
    # np.save(output / "feature_importances.npy", model.feature_importances_)

    stats['metrics'] = {}
    for part in X:
        p = predict(X[part])
        stats['metrics'][part] = lib.calculate_metrics(
            D.info['task_type'], Y[part], p, 'probs', y_info
        )
        # np.save(output / f'p_{part}.npy', p)
stats['time'] = lib.format_seconds(timer())
lib.dump_stats(stats, output, True)
lib.backup_output(output)
print(stats['metrics'])
# if 'downstream' in args['transfer']['stage']:
#     os.remove(output / 'checkpoint.pt')
