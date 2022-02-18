import toml
import os


# Defining parameters
models = ['catboost', 'xgboost']
dset_ids = ['california_housing', 41169, 'jannis', 1596, 'year', 41166, 554, 541, 40975, 1483, 42726, 42728, 40687, 188, 43172]
big_dsets = ['jannis', 1596, 'year', 41166, 554]
multiclass_dsets = [41169, 'jannis', 1596, 41166, 554, 40975, 1483, 40687, 188]
regression_dsets = ['california_housing', 'year', 541, 42726, 42728, 43172]

out_dir = f"output/optuning_configs/boost"
os.makedirs(out_dir, exist_ok=True)
launch = {}
for ds in dset_ids:
    launch[ds] = ''


def get_model_config(model, dset_big):

    if model == 'catboost':

        base_config_data = {
            "cat_policy": 'indices',
        }
        base_config_fit = {
            "logging_level": 'Verbose'
        }

        base_config_model = {
            "early_stopping_rounds": 50,
            "iterations": 2000,
            "metric_period": 10,
            "od_pval": 0.001,
            "task_type": 'GPU',
            "thread_count": 1,
        }

        optimization_space_model = {
            "bagging_temperature": [ 'uniform', 0.0, 1.0 ],
            "depth": [ 'int', 3, 10 ],
            "l2_leaf_reg": [ 'loguniform', 1.0, 10.0 ],
            "leaf_estimation_iterations": [ 'int', 1, 10 ],
            "learning_rate": [ 'loguniform', 1e-05, 1 ],
        }
        if dset_big:
            optimization_space_model['depth'] = [ 'int', 6, 10 ]

    elif model == 'xgboost':
        base_config_data = {
            "cat_policy": 'ohe',
        }

        base_config_fit ={
            "early_stopping_rounds": 50,
            "verbose": False
        }

        base_config_model = {
            "booster": 'gbtree',
            "n_estimators": 2000,
            "n_jobs": -1,
            "tree_method": 'gpu_hist',
        }

        optimization_space_model = {
            "alpha": [ '?loguniform', 0, 1e-08, 100.0 ],
            "colsample_bylevel": [ 'uniform', 0.5, 1.0 ],
            "colsample_bytree": [ 'uniform', 0.5, 1.0 ],
            "gamma": [ '?loguniform', 0, 1e-08, 100.0 ],
            "lambda": [ '?loguniform', 0, 1e-08, 100.0 ],
            "learning_rate": [ 'loguniform', 1e-05, 1 ],
            "max_depth": [ 'int', 3, 10 ],
            "min_child_weight": [ 'loguniform', 1e-08, 100000.0 ],
            "subsample": [ 'uniform', 0.5, 1.0 ],
        }

        if dset_big:
            optimization_space_model['max_depth'] = [ 'int', 6, 10 ]



    #################################################### Training args same for all deep models ######################

    base_config_transfer = {
        "stage": 'pretrain',
        "load_checkpoint": False,
        "checkpoint_path": float("nan"),
        "pretrain_proportion": 0.5,
        "downstream_samples_per_class": float("nan"),
        "freeze_feature_extractor": False,
        "layers_to_fine_tune": [],
        "use_mlp_head": False,
        "epochs_warm_up_head": 0,
        "head_lr":float("nan")
    }

    optimization_options = {
        "n_trials": 50
    }

    optimization_sampler = {
    "seed": 0
    }
    return base_config_data, base_config_model, base_config_fit, base_config_transfer, optimization_space_model, \
           optimization_sampler, optimization_options


for model in models:

    for dset_id in dset_ids:
        dset_is_big = dset_id in big_dsets
        base_config_data, base_config_model, base_config_fit, base_config_transfer, optimization_space_model, \
            optimization_sampler, optimization_options = get_model_config(model, dset_id)


        base_config_data['dset_id'] = dset_id
        if dset_id in multiclass_dsets:
            base_config_data['task'] = 'multiclass'
        elif dset_id in regression_dsets:
            base_config_data['task'] = 'regression'
            base_config_data['y_policy'] = 'mean_std'
        else:
            raise NameError('Dset_id is not known')

        if dset_id in ['aloi', 41166]:
            optimization_options["n_trials"] = 30


        toml_dict = {"program": 'bin/' + model + '_.py'}
        toml_dict['base_config'] = {
            "seed": 0,
            "data": base_config_data,
            "model": base_config_model,
            "fit": base_config_fit,
            "transfer": base_config_transfer
        }

        toml_dict['optimization'] = {
            "options": optimization_options,
            "sampler": optimization_sampler,
            "space": {"model": optimization_space_model}
        }


        with open(os.path.join(out_dir, f"optuning_{dset_id}_{model}.toml"), "w") as fh:
            toml.dump(toml_dict, fh)

        launch_str = f"python -W ignore bin/tune.py output/optuning_configs/boost/optuning_{dset_id}_{model}.toml -f \n"
        launch[dset_id] += launch_str

        os.makedirs('launch/optuna_jobs/boost', exist_ok=True)
        with open(f"launch/optuna_jobs/boost/optuning_{dset_id}_{model}.sh", "w") as fh:
            fh.write(launch_str)