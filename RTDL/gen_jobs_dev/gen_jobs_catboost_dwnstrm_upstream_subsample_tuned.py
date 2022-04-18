import toml
import os
import numpy as np

# Defining parameters
seeds = 5
models = ['catboost_mimic']
dset_ids = ['mimic']
downstream_targets = [0,1,4,5]
# n_estimators = [1000]
out_dir = f"output_mimic_dev/catboost_dwnstrm_upstream_subsample_tuned"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)
time = 1
global_launch_str = "\n"
for model in models:
    for downstream_target in downstream_targets:
        for dset_id in dset_ids:


            data_args = {'cat_policy': "indices",
                        'dset_id': dset_id,
                        'task': "binclass"}
            model_args = {'iterations': 1000,
                          'task_type': 'GPU',
                          'thread_count': 10,
                          'early_stopping_rounds': 50,
                          'od_pval': 0.001,
                          'metric_period': 10}
            fit_args = {'logging_level':'Verbose'}
            transfer_args = {'checkpoint_path' : np.nan,
                            'downstream_samples_per_class' : np.nan,
                            'epochs_warm_up_head' : 0,
                            'freeze_feature_extractor' : False,
                            'head_lr' : np.nan,
                            'layers_to_fine_tune' : [],
                            'load_checkpoint' : False,
                            'pretrain_proportion' : downstream_target,
                            'stage' : "pretrain",
                            'use_mlp_head' : False,
                            'use_optuna_CV': False,
                            'pretrain_subsample': False}



            for seed in range(seeds):
                data_args["dset_id"] = dset_id
                ################################################
                # Upstream pretraining:

                # standard supervised pretraining:
                toml_dict = {"seed": seed,
                             "data": data_args.copy(),
                             "model": model_args.copy(),
                             "fit": fit_args.copy(),
                             "transfer": transfer_args.copy()}
                toml_dict["model"]['task_type'] = 'CPU'
                with open(os.path.join(out_dir,
                                       f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}.toml"),
                          "w") as fh:
                    toml.dump(toml_dict, fh)

                launch_str = f"python -W ignore bin/{model}_.py {out_dir}/{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}.toml -f \n"
                for sample_num in sample_nums:
                    # load optuna best config
                    optuna_config_path = f"{os.path.dirname(out_dir)}/catboost_upstream_subsample_tuning/optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}/best.toml"
                    optuna_args = toml.load(optuna_config_path)

                    # load data args
                    # model_args = optuna_args["model"]
                    # fit_args = optuna_args["fit"]

                    ################################################
                    # Downstream baselines:
                    # only load optuna args for downstream jobs, do pretraining with defaults

                    #Turn on stacking
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": optuna_args["model"].copy(),
                                 "fit": optuna_args["fit"].copy(),
                                 "transfer": transfer_args.copy()}

                    toml_dict["model"]["use_best_model"] = False
                    toml_dict["model"]["early_stopping_rounds"] = None
                    toml_dict["model"]["od_pval"] = None
                    toml_dict["model"]["metric_period"] = None
                    toml_dict["transfer"]["checkpoint_path"] = f"{out_dir}/{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/model.cbm"
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]['load_checkpoint'] = True
                    with open(os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_upstream_subsample_tuned_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)


                    launch_str += f"python -W ignore bin/{model}_.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_upstream_subsample_tuned_seed{seed}.toml -f \n"

                os.makedirs('launch_mimic_dev/catboost_dwnstrm_upstream_subsample_tuned', exist_ok=True)
                with open(f"launch_mimic_dev/catboost_dwnstrm_upstream_subsample_tuned/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh", "w") as fh:
                    fh.write(launch_str)
                global_launch_str += f"python3 dispatch_single_node.py launch_mimic_dev/catboost_dwnstrm_upstream_subsample_tuned/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh --name {dset_id}{downstream_target}_{model}_exp_seed{seed} --time {time} \n"

os.makedirs('launch_mimic_dev/global_launch', exist_ok=True)
with open(f"launch_mimic_dev/global_launch/catboost_dwnstrm_upstream_subsample_tuned.sh", "w") as fh:
    fh.write(global_launch_str)