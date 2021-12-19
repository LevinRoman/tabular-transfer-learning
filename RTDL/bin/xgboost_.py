from copy import deepcopy
from pathlib import Path

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

N, C, y, info = lib.data_prep_openml(ds_id = dset_id, seed = args['seed'], task = args['data']['task'], datasplit=[.65, .15, .2])
D = lib.Dataset(N, C, y, info)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'ohe'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
assert isinstance(X, dict)
zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')


fit_kwargs = deepcopy(args["fit"])
# XGBoost does not automatically use the best model, so early stopping must be used
assert 'early_stopping_rounds' in fit_kwargs
fit_kwargs['eval_set'] = [(X[lib.VAL], Y[lib.VAL])]
if D.is_regression:
    model = XGBRegressor(**args["model"])
    predict = model.predict
else:
    model = XGBClassifier(**args["model"], disable_default_eval_metric=True)
    if D.is_multiclass:
        predict = model.predict_proba
        fit_kwargs['eval_metric'] = 'merror'
    else:
        predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
        fit_kwargs['eval_metric'] = 'error'

# Fit model
timer = zero.Timer()
timer.run()
model.fit(X[lib.TRAIN], Y[lib.TRAIN], **fit_kwargs)

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
