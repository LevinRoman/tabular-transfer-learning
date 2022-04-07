import toml
import os
import numpy as np

# Defining parameters
seeds = 5
models = ['catboost_mimic'] #ft_transformer_mlm for Avi's things
dset_ids = ['mimic']
downstream_targets = list(range(12))
n_estimators = [100]
time = 10
out_dir = f"output/optuning_configs/tune_upstream_catboost"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)

def get_model_config(model, dset_big=False):

    if 'catboost' in model:

        base_config_data = {
            "cat_policy": 'indices',
        }
        base_config_fit = {
            "logging_level": 'Verbose'
        }

        base_config_model = {
            # "early_stopping_rounds": False,
            # "iterations": 1000,
            "metric_period": 10000,
            "od_pval": 0,
            "task_type": 'GPU',
            "thread_count": 1,
            "use_best_model": False
        }

        optimization_space_model = {
            "iterations": ['int', 2, 1000],
            "bagging_temperature": [ 'uniform', 0.0, 1.0 ],
            "depth": [ 'int', 3, 10 ],
            "l2_leaf_reg": [ 'loguniform', 1.0, 10.0 ],
            "leaf_estimation_iterations": [ 'int', 1, 10 ],
            "learning_rate": [ 'loguniform', 1e-05, 1 ],
        }
        if dset_big:
            optimization_space_model['depth'] = [ 'int', 6, 10 ]

    elif 'xgboost' in model:
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
        "head_lr":float("nan"),
        "use_optuna_CV": False,
        "pretrain_subsample": True
    }

    optimization_options = {
        "n_trials": 100
    }

    optimization_sampler = {
    "seed": 0
    }
    return base_config_data, base_config_model, base_config_fit, base_config_transfer, optimization_space_model, \
           optimization_sampler, optimization_options


global_launch_str = "\n"
for model in models:
    for downstream_target in downstream_targets:
        for sample_num in sample_nums:
            for dset_id in dset_ids:
                #TODO: load optuned configs when optuna is done for upstream

                base_config_data, base_config_model, base_config_fit, base_config_transfer, optimization_space_model, \
                optimization_sampler, optimization_options = get_model_config(model, dset_id)
                # optuna_config_path = f"../RTDL/output/optuning_configs/{dset_id}/{model}/0/best.toml"
                # optuna_args = toml.load(optuna_config_path)
                base_config_data['dset_id'] = dset_id
                base_config_data['task'] = 'binclass'
                base_config_transfer['pretrain_proportion'] = downstream_target
                base_config_transfer['downstream_samples_per_class'] = sample_num
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

                with open(os.path.join(out_dir, f"optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

                launch_str = f"python -W ignore bin/tune.py {out_dir}/optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.toml -f \n"

                os.makedirs('launch/optuna_jobs/tune_upstream_catboost', exist_ok=True)
                with open(f"launch/optuna_jobs/tune_upstream_catboost/optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.sh", "w") as fh:
                    fh.write(launch_str)
                global_launch_str += f"python3 dispatch_single_node.py launch/optuna_jobs/tune_upstream_catboost/optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.sh --name optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model} --time {time} \n"

with open(f"launch/mimic/mimic_tune_upstream_catboost.sh", "w") as fh:
    fh.write(global_launch_str)