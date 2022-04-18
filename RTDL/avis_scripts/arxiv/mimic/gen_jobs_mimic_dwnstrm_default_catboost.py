import toml
import os
import numpy as np

# Defining parameters
seeds = 5
models = ['catboost_mimic'] #ft_transformer_mlm for Avi's things
dset_ids = ['mimic']
downstream_targets = list(range(12))
n_estimators = [100]
out_dir = f"output/default_dwnstrm_catboost_100trees"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)
time = 1
global_launch_str = "\n"
for model in models:
    for downstream_target in downstream_targets:
        for dset_id in dset_ids:


            data_args = {'cat_policy': "indices",
                        'dset_id': dset_id,
                        'normalization': "quantile",
                        'task': "binclass"}
            model_args = {'iterations':1000, 'task_type': 'GPU', 'thread_count':10}
            fit_args = {'logging_level':'Verbose'}
            transfer_args = {'checkpoint_path' : np.nan,
                            'downstream_samples_per_class' : 50,
                            'epochs_warm_up_head' : 0,
                            'freeze_feature_extractor' : False,
                            'head_lr' : np.nan,
                            'layers_to_fine_tune' : [],
                            'load_checkpoint' : False,
                            'pretrain_proportion' : downstream_target,
                            'stage' : "pretrain",
                            'use_mlp_head' : False,
                            'use_optuna_CV': False }



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
                    for n_est in n_estimators:
                        ################################################
                        # Downstream baselines:

                        #Turn on stacking
                        toml_dict = {"seed": seed,
                                     "data": data_args.copy(),
                                     "model": model_args.copy(),
                                     "fit": fit_args.copy(),
                                     "transfer": transfer_args.copy()}
                        toml_dict["model"]["iterations"] = n_est
                        toml_dict["model"]["use_best_model"] = False
                        toml_dict["model"]["early_stopping_rounds"] = None
                        toml_dict["model"]["od_pval"] = None
                        toml_dict["transfer"]["checkpoint_path"] = f"{out_dir}/{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/model.cbm"
                        toml_dict["transfer"]["stage"] = "downstream"
                        toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                        toml_dict["transfer"]['load_checkpoint'] = True
                        with open(os.path.join(out_dir,
                                               f"{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_nestimators{n_est}_seed{seed}.toml"),
                                  "w") as fh:
                            toml.dump(toml_dict, fh)


                        launch_str += f"python -W ignore bin/{model}_.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_nestimators{n_est}_seed{seed}.toml -f \n"

                os.makedirs('launch/default_catboost_jobs_100trees', exist_ok=True)
                with open(f"launch/default_catboost_jobs_100trees/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh", "w") as fh:
                    fh.write(launch_str)
                global_launch_str += f"python3 dispatch_single_node.py launch/default_catboost_jobs_100trees/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh --name {dset_id}{downstream_target}_{model}_exp_seed{seed} --time {time} \n"

with open(f"launch/mimic/mimic_dwnstrm_default_catboost_100trees.sh", "w") as fh:
    fh.write(global_launch_str)