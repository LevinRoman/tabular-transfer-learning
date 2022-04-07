import os
import shutil
from pathlib import Path
# previous catboost version: '0.24.4'
import numpy as np
import pandas as pd
import zero
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
import lib
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, recall_score
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
args, output = lib.load_config()
args['model']['random_seed'] = args['seed']
assert (
    'task_type' in args['model']
)  # Significantly affects performance, so must be set explicitely
if args['model']['task_type'] == 'GPU':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')

zero.set_randomness(args['seed'])
dset_id = args['data']['dset_id']
stats = lib.load_json(output / 'stats.json')
stats.update({'dataset': dset_id, 'algorithm': Path(__file__).stem})

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


D = lib.Dataset(N, C, y, info)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'indices'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
    full_cat_data_for_encoder = full_cat_data_for_encoder
)

zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
# lib.dump_pickle(y_info, output / 'y_info.pickle')

model_kwargs = args['model']

if args['data'].get('cat_policy') == 'indices':
    assert isinstance(X, tuple)
    N, C, _, _ = X
    n_num_features = 0 if N is None else N[lib.TRAIN].shape[1]
    n_cat_features = 0 if C is None else C[lib.TRAIN].shape[1]
    n_features = n_num_features + n_cat_features
    if N is None:
        assert C is not None
        X = {x: pd.DataFrame(C[x], columns=range(n_features)) for x in C}
    elif C is None:
        assert N is not None
        X = {x: pd.DataFrame(N[x], columns=range(n_features)) for x in N}
    else:
        X = {
            k: pd.concat(
                [
                    pd.DataFrame(N[k], columns=range(n_num_features)),
                    pd.DataFrame(C[k], columns=range(n_num_features, n_features)),
                ],
                axis=1,
            )
            for k in N.keys()
        }

    model_kwargs['cat_features'] = list(range(n_num_features, n_features))


if model_kwargs['task_type'] == 'GPU':
    model_kwargs['devices'] = '0'
if D.is_regression:
    model = CatBoostRegressor(**model_kwargs, allow_writing_files = False)
    predict = model.predict
elif D.is_multiclass:
    model = CatBoostClassifier(**model_kwargs, eval_metric='Accuracy', allow_writing_files = False)
    predict = (
        model.predict_proba
        if D.is_multiclass
        else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
    )
else:
    if 'pretrain' in args['transfer']['stage']:
        if not args['transfer']['pretrain_subsample']:
            model = CatBoostClassifier(**model_kwargs,
                loss_function='MultiLogloss',
                eval_metric='HammingLoss',
                class_names=['class_{}'.format(i) for i in range(Y[lib.TRAIN].shape[1])],
                allow_writing_files=False
            )
            predict = model.predict_proba
        else:
            model = CatBoostClassifier(**model_kwargs, eval_metric='AUC', allow_writing_files=False, verbose=0)
            predict = lambda x: model.predict_proba(x)[:, 1]
    elif 'downstream' in args['transfer']['stage']:
        model = CatBoostClassifier(**model_kwargs, eval_metric='AUC', allow_writing_files=False, verbose=0)
        predict = lambda x: model.predict_proba(x)[:, 1]
    else:
        raise ValueError('stage can be only downstream or pretrain')

timer = zero.Timer()
timer.run()
if 'downstream' in args['transfer']['stage']:
    # loading checkpoint of the upstream model, do stacking
    if (args['transfer']['load_checkpoint']):
        print('Stacking')
        for part in X:
            from_file = CatBoostClassifier()
            from_file.load_model(args['transfer']['checkpoint_path'])
            upstream_predictions = from_file.predict(X[part])
            upstream_predictions = pd.DataFrame(upstream_predictions, columns=['upstream_'+str(i) for i in range(upstream_predictions.shape[1])])
            X[part] = pd.concat([X[part], upstream_predictions], axis = 1) #note that cat features are specified by index
    else:
        print('No stacking')
    # Tune downstream hyperparameters with optuna
    if args['transfer']['use_optuna_CV']:
        raise NotImplementedError('We dont use optuna CV!')
        cv_inner = StratifiedKFold(n_splits=min(X[lib.TRAIN].shape[0], 5), shuffle=True, random_state=args['seed'])
        param_distributions = lib.util.get_param_distributions('catboost')
        optuna_search = optuna.integration.OptunaSearchCV(model,
                                                          param_distributions,
                                                          scoring=sklearn.metrics.make_scorer(roc_auc_score),
                                                          refit=True,
                                                          cv=cv_inner,
                                                          random_state=args['seed'],
                                                          n_trials=100,
                                                          verbose=0,
                                                          error_score=np.nan
                                                          )
        optuna_search.fit(X[lib.TRAIN], Y[lib.TRAIN])
        print('Best score:', optuna_search.best_score_)
        model = optuna_search.best_estimator_
    else:
        model.fit(
            X[lib.TRAIN],
            Y[lib.TRAIN],
            **args['fit']
        )
else:
    model.fit(
        X[lib.TRAIN],
        Y[lib.TRAIN],
        **args['fit'],
        eval_set=(X[lib.VAL], Y[lib.VAL]),
    )
    model.save_model(str(output / 'model.cbm'))
if Path('catboost_info').exists():
    shutil.rmtree('catboost_info', ignore_errors=True)

# model.save_model(str(output / 'model.cbm'))
# np.save(output / 'feature_importances.npy', model.get_feature_importance())
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
