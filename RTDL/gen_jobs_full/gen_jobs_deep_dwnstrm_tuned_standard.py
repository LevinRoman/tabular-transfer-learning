import toml
import os


# Defining parameters
seeds = 5
models = ['ft_transformer', 'resnet', 'mlp', 'tab_transformer']#, 'saint']#'saint' #ft_transformer_mlm for Avi's things
dset_ids = ['mimic']
downstream_targets = [0,1,2,3,4,5,6,7,8,9,10,11]
big_dsets = []
multiclass_dsets = []
regression_dsets = []
binclass_dsets = ['mimic']


out_dir = f"output_mimic_full/deep_dwnstrm_tuned_standard"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)
time = 3
global_launch_str = "\n"
for model in models:
    for downstream_target in downstream_targets:
        for dset_id in dset_ids:
            #could change the path to output_mimic_full optuna after running it there
            optuna_config_path = f"output/optuning_configs/deep/optuning_{dset_id}_{model}_{downstream_target}/best.toml"
            optuna_args = toml.load(optuna_config_path)

            # load data args
            data_args = optuna_args["data"]
            model_args = optuna_args["model"]
            training_args = optuna_args["training"]
            transfer_args = optuna_args["transfer"]
            transfer_args['pretrain_subsample'] = False
            for seed in range(seeds):
                ################################################
                # Upstream pretraining:

                # standard supervised pretraining:
                toml_dict = {"seed": seed,
                             "data": data_args.copy(),
                             "model": model_args.copy(),
                             "training": training_args.copy(),
                             "transfer": transfer_args.copy()}
                with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

                launch_str = f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}.toml -f \n"


                for sample_num in sample_nums:
                    ################################################
                    # Downstream baselines:

                    # downstream supervised learning:
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["training"]["lr"] = 1e-4
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_seed{seed}.toml"), "w") as fh:
                        toml.dump(toml_dict, fh)

                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_seed{seed}.toml -f \n"

                    # # downstream random feature extractor learning:
                    # toml_dict = {"seed": seed,
                    #              "data": data_args.copy(),
                    #              "model": model_args.copy(),
                    #              "training": training_args.copy(),
                    #              "transfer": transfer_args.copy()}
                    # toml_dict["training"]["n_epochs"] = 200
                    # toml_dict["training"]["patience"] = 1e5
                    # toml_dict["training"]["lr"] = 1e-4
                    # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    # toml_dict["transfer"]["epochs_warm_up_head"] = 0
                    # toml_dict["transfer"]["freeze_feature_extractor"] = True
                    # toml_dict["transfer"]["head_lr"] = 1e-4
                    # toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                    # toml_dict["transfer"]["load_checkpoint"] = False
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["use_mlp_head"] = True
                    #
                    # with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_random_extractor_{model}_seed{seed}.toml"), "w") as fh:
                    #     toml.dump(toml_dict, fh)
                    #
                    # launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_random_extractor_{model}_seed{seed}.toml -f \n"


                    ################################################
                    # Downstream transfering from supervised pretraining:

                    # # downstream finetune linear head:
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["training"]["lr"] = 1e-4
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 0
                    toml_dict["transfer"]["freeze_feature_extractor"] = True
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = False
                    with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_linear_head_from_supervised_pretrain_{model}_seed{seed}.toml"), "w") as fh:
                        toml.dump(toml_dict, fh)

                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_linear_head_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"

                    # downstream finetune mlp head from supervised pretraining:
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["training"]["lr"] = 1e-4
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 0
                    toml_dict["transfer"]["freeze_feature_extractor"] = True
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = True
                    with open(os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_mlp_head_from_supervised_pretrain_{model}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_mlp_head_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"


                    # downstream finetune whole model with linear head from supervised pretraining:
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["lr"] = 1e-4 / 2
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,  f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    toml_dict["transfer"]["freeze_feature_extractor"] = False
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = []
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = False
                    with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml"), "w") as fh:
                        toml.dump(toml_dict, fh)

                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"


                    # downstream finetune whole model with mlp head:
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["training"]["lr"] = 1e-4 / 2
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    toml_dict["transfer"]["freeze_feature_extractor"] = False
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = []
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = True
                    with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_mlp_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml"), "w") as fh:
                        toml.dump(toml_dict, fh)

                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_mlp_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"

                supervised_pretraining_path_checkpoint = os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}", 'checkpoint.pt')

                launch_str += f"python destructive_danger/delete_checkpoint.py --file_root {supervised_pretraining_path_checkpoint}\n"
                os.makedirs('launch_mimic_full/deep_dwnstrm_tuned_standard', exist_ok=True)
                with open(f"launch_mimic_full/deep_dwnstrm_tuned_standard/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh", "w") as fh:
                    fh.write(launch_str)
                global_launch_str += f"python3 dispatch_single_node.py launch_mimic_full/deep_dwnstrm_tuned_standard/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh --name {dset_id}{downstream_target}_{model}_exp_seed{seed} --time {time} \n"

os.makedirs('launch_mimic_full/global_launch', exist_ok=True)
with open(f"launch_mimic_full/global_launch/deep_dwnstrm_tuned_standard.sh", "w") as fh:
    fh.write(global_launch_str)