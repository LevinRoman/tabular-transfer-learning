import toml
import os


# Defining parameters
seeds = 5
models = ['ft_transformer']#, 'resnet']#, 'mlp', 'tab_transformer']
dset_ids = ['mimic']
#could subsample targets, probably don't need all for imputation experiments, will share good targets a bit later
downstream_targets = [0,2,8]

#create output dir for imputation experiments
out_dir = f"output_mimic_imputation/deep_dwnstrm_tuned_imputation"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)

#Adjust the time for cml jobs if needed
time = 8
global_launch_str = "\n"
for model in models:
    for downstream_target in downstream_targets:
        for dset_id in dset_ids:
            #path to optuned configs, double check that this is where optuned configs are stored
            optuna_config_path = f"optuning_configs/optuning_{dset_id}_{model}_{downstream_target}/best.toml"
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
                ### This is upstream with a missing column
                toml_dict['transfer']['column_mode'] = 'remove_column'

                #save config in the output directory
                with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_with_missing_column_{model}_seed{seed}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)
                #add the line to run this config to the launch string which we will dump into a .sh file later
                launch_str = f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_supervised_pretraining_with_missing_column_{model}_seed{seed}.toml -f \n"





                # upstream + train_to_predict_missing_column used in second type of imputation

                toml_dict = {"seed": seed,
                             "data": data_args.copy(),
                             "model": model_args.copy(),
                             "training": training_args.copy(),
                             "transfer": transfer_args.copy()}

                toml_dict["transfer"]["checkpoint_path"] = None
                toml_dict['transfer']['column_mode'] = 'train_to_predict_missing_column'
                toml_dict['data']['task'] = 'regression'
                toml_dict['data']['y_policy'] = 'mean_std'

                with open(os.path.join(out_dir,
                                       f"{dset_id}{downstream_target}_upstream_train_to_predict_missing_column_{model}_seed{seed}.toml"),
                          "w") as fh:
                    toml.dump(toml_dict, fh)
                # add the line to run this config to the launch string which we will dump into a .sh file later
                launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_upstream_train_to_predict_missing_column_{model}_seed{seed}.toml -f \n"



                # normal upstreme training. Used in second type of imputation
                toml_dict = {"seed": seed,
                             "data": data_args.copy(),
                             "model": model_args.copy(),
                             "training": training_args.copy(),
                             "transfer": transfer_args.copy()}

                toml_dict["transfer"]["column_mode"] = "None"
                # save config in the output directory
                with open(os.path.join(out_dir,
                                       f"{dset_id}{downstream_target}_supervised_pretraining_with_all_column_{model}_seed{seed}.toml"),
                          "w") as fh:
                    toml.dump(toml_dict, fh)
                # add the line to run this config to the launch string which we will dump into a .sh file later
                launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_supervised_pretraining_with_all_column_{model}_seed{seed}.toml -f \n"



                for sample_num in sample_nums:
                    ################################################
                    # Downstream baselines: commented out, not needed for imputation experiments

                    # downstream supervised learning:
                    # toml_dict = {"seed": seed,
                    #              "data": data_args.copy(),
                    #              "model": model_args.copy(),
                    #              "training": training_args.copy(),
                    #              "transfer": transfer_args.copy()}
                    # toml_dict["training"]["n_epochs"] = 200
                    # toml_dict["training"]["patience"] = 1e5
                    # toml_dict["training"]["lr"] = 1e-4
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    # with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_seed{seed}.toml"), "w") as fh:
                    #     toml.dump(toml_dict, fh)
                    #
                    # launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_fromscratch_{model}_seed{seed}.toml -f \n"

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
                    # Downstream transfering from supervised pretraining: 4 default setups
                    # *linear head + frozen feat extractor
                    # *mlp head + frozen feat extractor
                    # *linear head + whole model fine-tuned
                    # *mlp head + whole model fine-tuned
                    #for imputation experiments, we probably can focus on just linear head + whole model fine-tuned
                    #all other setups are commented out below
                    #TODO: add imputation-specific arguments and setups (e.g. using missing/imputed/ground-truth feature)

                    # # downstream finetune just linear head from supervised pretraining, frozen feature extractor:
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
                    # toml_dict["transfer"]["load_checkpoint"] = True
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["use_mlp_head"] = False
                    # with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_linear_head_from_supervised_pretrain_{model}_seed{seed}.toml"), "w") as fh:
                    #     toml.dump(toml_dict, fh)
                    #
                    # launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_linear_head_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"
                    #
                    # # downstream finetune just mlp head from supervised pretraining, frozen feature extractor:
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
                    # toml_dict["transfer"]["load_checkpoint"] = True
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["use_mlp_head"] = True
                    # with open(os.path.join(out_dir,
                    #                        f"{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_mlp_head_from_supervised_pretrain_{model}_seed{seed}.toml"),
                    #           "w") as fh:
                    #     toml.dump(toml_dict, fh)
                    # launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_tuned_mlp_head_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"


                    # downstream finetune whole model with linear head from supervised pretraining:
                    #this is the default setup, where all features are available
                    #might need to add an argument about that

                    #sample_nums = [2, 5, 10, 50, 100]
                    if sample_num in [2, 5]:
                        imputation_epoch = 30
                    elif sample_num in [10]:
                        imputation_epoch = 60
                    elif sample_num in [50, 100]:
                        imputation_epoch = 75

                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["lr"] = 1e-4 / 2
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,  f"{dset_id}{downstream_target}_supervised_pretraining_with_missing_column_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    toml_dict["transfer"]["freeze_feature_extractor"] = False
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = []
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = False
                    toml_dict['transfer']['column_mode'] = 'remove_column'

                    #save config
                    with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_with_missing_column_{model}_seed{seed}.toml"), "w") as fh:
                        toml.dump(toml_dict, fh)
                    # add the line to run this config to the launch string which we will dump into a .sh file later
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_with_missing_column_{model}_seed{seed}.toml -f \n"




                    # downstream + train_to_predict_missing_column
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["lr"] = 1e-4 / 2
                    toml_dict["training"]["n_epochs"] = imputation_epoch
                    toml_dict["training"]["patience"] = 1e5

                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    toml_dict["transfer"]["freeze_feature_extractor"] = False
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = []
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = False


                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,  f"{dset_id}{downstream_target}_supervised_pretraining_with_missing_column_{model}_seed{seed}/checkpoint.pt")
                    toml_dict['transfer']['column_mode'] = 'train_to_predict_missing_column'
                    toml_dict['data']['task'] = 'regression'
                    toml_dict['data']['y_policy'] = 'mean_std'


                    with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_train_to_predict_missing_column_{model}_seed{seed}.toml"), "w") as fh:
                        toml.dump(toml_dict, fh)
                    # add the line to run this config to the launch string which we will dump into a .sh file later
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_train_to_predict_missing_column_{model}_seed{seed}.toml -f \n"


                    # upstream + predict_missing_column
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}

                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_train_to_predict_missing_column_{model}_seed{seed}/checkpoint.pt")
                    toml_dict['transfer']['column_mode'] = 'predict_missing_column'
                    toml_dict['data']['task'] = 'regression'
                    toml_dict['data']['y_policy'] = 'mean_std'

                    ## added this to have the info that what is the number of downstream samples being used for imputation stuff
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num

                    with open(os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_upstream_{sample_num}samples_predict_missing_column_{model}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)
                    # add the line to run this config to the launch string which we will dump into a .sh file later
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_upstream_{sample_num}samples_predict_missing_column_{model}_seed{seed}.toml -f \n"



                    # upstream + train_with_missing_column
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}

                    toml_dict['transfer']['column_mode'] = 'train_with_imputed_column'

                    ## added this to have the info that what is the number of downstream samples being used for imputation stuff
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num

                    with open(os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_upstream_{sample_num}samples_train_with_imputed_column_{model}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)
                    # add the line to run this config to the launch string which we will dump into a .sh file later
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_upstream_{sample_num}samples_train_with_imputed_column_{model}_seed{seed}.toml -f \n"



                    #downstream + None + transfer just use the above upstream check point
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}
                    toml_dict["training"]["lr"] = 1e-4 / 2
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                                                                            f"{dset_id}{downstream_target}_upstream_{sample_num}samples_train_with_imputed_column_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    toml_dict["transfer"]["freeze_feature_extractor"] = False
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = []
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = False
                    toml_dict["transfer"]["column_mode"] = "None"

                    # save config
                    with open(os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_with_imputed_column_{model}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)
                    # add the line to run this config to the launch string which we will dump into a .sh file later
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_with_imputed_column_{model}_seed{seed}.toml -f \n"



                    ######## second type of imputation #########


                    # downstream + predict_missing_column

                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}

                    toml_dict["training"]["lr"] = 1e-4 / 2
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_upstream_train_to_predict_missing_column_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    toml_dict["transfer"]["freeze_feature_extractor"] = False
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = []
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = False
                    toml_dict['transfer']['column_mode'] = 'predict_missing_column'
                    toml_dict['data']['task'] = 'regression'
                    toml_dict['data']['y_policy'] = 'mean_std'


                    with open(os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_downstream_{sample_num}samples_predict_missing_column_{model}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)
                    # add the line to run this config to the launch string which we will dump into a .sh file later
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_predict_missing_column_{model}_seed{seed}.toml -f \n"

                    #downstream + train_with_missing_column + transfer

                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "training": training_args.copy(),
                                 "transfer": transfer_args.copy()}

                    toml_dict["training"]["lr"] = 1e-4 / 2
                    toml_dict["training"]["n_epochs"] = 200
                    toml_dict["training"]["patience"] = 1e5
                    toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_supervised_pretraining_with_all_column_{model}_seed{seed}/checkpoint.pt")
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    toml_dict["transfer"]["freeze_feature_extractor"] = False
                    toml_dict["transfer"]["head_lr"] = 1e-4
                    toml_dict["transfer"]["layers_to_fine_tune"] = []
                    toml_dict["transfer"]["load_checkpoint"] = True
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["use_mlp_head"] = False
                    toml_dict['transfer']['column_mode'] = 'train_with_imputed_column'
                    toml_dict['transfer']['downstream_target'] = downstream_target


                    with open(os.path.join(out_dir,
                                           f"{dset_id}{downstream_target}_downstream_{sample_num}samples_train_with_imputed_column_{model}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)
                    # add the line to run this config to the launch string which we will dump into a .sh file later
                    launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_train_with_imputed_column_{model}_seed{seed}.toml -f \n"



                    #TODO: add a setup with missing upstream feature
                    #TODO: add a setup with imputed upstream feature

                    #TODO: add a setup with missing downstream feature
                    #TODO: add a setup with imputed downstream feature

                    # # downstream finetune whole model with mlp head from supervised pretraining:
                    # toml_dict = {"seed": seed,
                    #              "data": data_args.copy(),
                    #              "model": model_args.copy(),
                    #              "training": training_args.copy(),
                    #              "transfer": transfer_args.copy()}
                    # toml_dict["training"]["n_epochs"] = 200
                    # toml_dict["training"]["patience"] = 1e5
                    # toml_dict["training"]["lr"] = 1e-4 / 2
                    # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    # toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    # toml_dict["transfer"]["freeze_feature_extractor"] = False
                    # toml_dict["transfer"]["head_lr"] = 1e-4
                    # toml_dict["transfer"]["layers_to_fine_tune"] = []
                    # toml_dict["transfer"]["load_checkpoint"] = True
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["use_mlp_head"] = True
                    # with open(os.path.join(out_dir, f"{dset_id}{downstream_target}_downstream_{sample_num}samples_mlp_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml"), "w") as fh:
                    #     toml.dump(toml_dict, fh)
                    #
                    # launch_str = launch_str + f"python -W ignore bin/{model}.py {out_dir}/{dset_id}{downstream_target}_downstream_{sample_num}samples_mlp_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"

                supervised_pretraining_path_checkpoint = os.path.join(out_dir, f"{dset_id}{downstream_target}_supervised_pretraining_{model}_seed{seed}", 'checkpoint.pt')
                #add a line to delete the pretrained checkpoint to save space (they could be huge)
                launch_str += f"python destructive_danger/delete_checkpoint.py --file_root {supervised_pretraining_path_checkpoint}\n"
                os.makedirs('launch_mimic_imputation/deep_dwnstrm_tuned_imputation', exist_ok=True)
                #dump the launch string into .sh in the launch directory
                with open(f"launch_mimic_imputation/deep_dwnstrm_tuned_imputation/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh", "w") as fh:
                    fh.write(launch_str)
                #augment the global launch string which would dispatch all experiments
                global_launch_str += f"python3 dispatch_single_node.py launch_mimic_imputation/deep_dwnstrm_tuned_imputation/{dset_id}{downstream_target}_{model}_exp_seed{seed}.sh --name {dset_id}{downstream_target}_{model}_exp_seed{seed} --time {time} \n"

#save the global launch string
os.makedirs('launch_mimic_imputation/global_launch', exist_ok=True)
with open(f"launch_mimic_imputation/global_launch/deep_dwnstrm_tuned_imputation.sh", "w") as fh:
    fh.write(global_launch_str)
#launching all experiments is then:
#chmod +x launch_mimic_imputation/global_launch/deep_dwnstrm_tuned_imputation.sh
#./launch_mimic_imputation/global_launch/deep_dwnstrm_tuned_imputation.sh