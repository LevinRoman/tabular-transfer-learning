import toml
import os


# Defining parameters
seeds = 5
models = ['resnet', 'mlp', 'saint', 'ft_transformer', 'tab_transformer'] #ft_transformer_mlm for Avi's things
dset_ids = ['california_housing', 41169, 'jannis', 1596, 'year', 541, 40975, 1483, 42726, 42728, 40687, 188, 43172] #554, 'aloi', 41166,
big_dsets = ['jannis', 1596, 'year', 'aloi', 41166, 554]
multiclass_dsets = [41169, 'jannis', 1596, 'aloi', 41166, 554, 40975, 1483, 40687, 188]
regression_dsets = ['california_housing', 'year', 541, 42726, 42728, 43172]


out_dir = f"output/default_dwnstrm"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)


def get_default_config(model, config):
    if model == 'ft_transformer':
        config['model']['attention_dropout'] = 0.2
        config['model']['d_ffn_factor'] = 4/3
        config['model']['d_token'] = 192
        config['model']['ffn_dropout'] = 0.1
        config['model']['n_heads'] = 8
        config['model']['n_layers'] = 3
        config['model']['residual_dropout'] = 0.0
        config['training']['lr'] = 0.0001
        config['training']['weight_decay'] = 1e-5

    elif model == 'saint':
        config['model']['embed_dim'] = 32
        config['model']['heads'] = 8
        config['model']['attn_dropout'] = 0.8
        config['model']['ff_dropout'] = 0.8
        config['training']['lr'] = 0.0001
        config['training']['weight_decay'] = 1e-5

    elif model == 'resnet':
        config['model']['d'] = 200
        config['model']['d_embedding'] = 128
        config['model']['d_hidden_factor'] = 3
        config['model']['hidden_dropout'] = 0.2
        config['model']['n_layers'] = 5
        config['model']['residual_dropout'] = 0.2
        config['training']['lr'] = 0.0001
        config['training']['weight_decay'] = 0.0

    elif model == 'mlp':
        config['model']['d_layers'] = [ 300, 200, 300 ]
        config['model']['dropout'] = 0.2
        config['model']['d_embedding'] = 32
        config['training']['lr'] = 0.0001
        config['training']['weight_decay'] = 1e-5

    elif model == 'tab_transformer':
        config['model']['dim'] = 32
        config['model']['depth'] = 6
        config['model']['heads'] = 8
        config['model']['dim_heads'] = 16
        config['model']['attn_dropout'] = 0.0
        config['model']['ff_dropout'] = 0.0
        config['training']['lr'] = 0.0001
        config['training']['weight_decay'] = 1e-5
    else:
        raise NameError('Model is not defined')

    return config






for model in models:
    for dset_id in dset_ids:
        optuna_config_path = f"output/optuning_configs/deep/optuning_{dset_id}_{model}/best.toml"
        optuna_args = toml.load(optuna_config_path)

        optuna_args_default = get_default_config(model, optuna_args)

        # load data args
        data_args = optuna_args_default["data"]
        model_args = optuna_args_default["model"]
        training_args = optuna_args_default["training"]
        transfer_args = optuna_args_default["transfer"]

        for seed in range(seeds):
            ################################################
            # Upstream pretraining:

            # standard supervised pretraining:
            toml_dict = {"seed": seed,
                         "data": data_args.copy(),
                         "model": model_args.copy(),
                         "training": training_args.copy(),
                         "transfer": transfer_args.copy()}
            with open(os.path.join(out_dir, f"{dset_id}_supervised_pretraining_{model}_seed{seed}.toml"), "w") as fh:
                toml.dump(toml_dict, fh)

            # # mlm + supervised pretraining:
            # toml_dict = {"seed": seed,
            #              "data": data_args.copy(),
            #              "model": model_args.copy(),
            #              "training": training_args.copy(),
            #              "transfer": transfer_args.copy()}
            # toml_dict["training"]["mlm_masking_proportion"] = 0.15
            # toml_dict["training"]["mlm_loss_masked_only"] = True
            # toml_dict["training"]["mlm_mode"] = "mlm supervised"
            # toml_dict["training"]["num_batch_warm_up"] = 0
            # toml_dict["training"]["self_supervised"] = True
            # toml_dict["training"]["sf_loss_lam"] = 0.001
            # with open(os.path.join(out_dir, f"{dset_id}_mlm_pretraining_ft-trans_seed{seed}.toml"), "w") as fh:
            #     toml.dump(toml_dict, fh)

            # launch_str = f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_supervised_pretraining_ft-trans_seed{seed}.toml -f \n" \
            #              f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_mlm_pretraining_ft-trans_seed{seed}.toml -f \n"
            launch_str = f"python -W ignore bin/{model}.py output/optuna_dwnstrm/{dset_id}_supervised_pretraining_{model}_seed{seed}.toml -f \n"


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
                with open(os.path.join(out_dir, f"{dset_id}_downstream_{sample_num}samples_fromscratch_{model}_seed{seed}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

                # downstream random feature extractor learning:
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
                toml_dict["transfer"]["freeze_feature_extractor"] = True
                toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                with open(os.path.join(out_dir, f"{dset_id}_downstream_{sample_num}samples_random_extractor_{model}_seed{seed}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

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
                toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                                                                        f"{dset_id}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                toml_dict["transfer"]["epochs_warm_up_head"] = 0
                toml_dict["transfer"]["freeze_feature_extractor"] = True
                toml_dict["transfer"]["head_lr"] = 1e-4
                toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                toml_dict["transfer"]["load_checkpoint"] = True
                toml_dict["transfer"]["stage"] = "downstream"
                toml_dict["transfer"]["use_mlp_head"] = False
                with open(os.path.join(out_dir, f"{dset_id}_downstream_{sample_num}samples_tuned_linear_head_from_supervised_pretrain_{seed}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

                # downstream finetune mlp head from supervised pretraining:
                toml_dict = {"seed": seed,
                             "data": data_args.copy(),
                             "model": model_args.copy(),
                             "training": training_args.copy(),
                             "transfer": transfer_args.copy()}
                toml_dict["training"]["n_epochs"] = 200
                toml_dict["training"]["patience"] = 1e5
                toml_dict["training"]["lr"] = 1e-4
                toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                                                                        f"{dset_id}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                toml_dict["transfer"]["epochs_warm_up_head"] = 0
                toml_dict["transfer"]["freeze_feature_extractor"] = True
                toml_dict["transfer"]["head_lr"] = 1e-4
                toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                toml_dict["transfer"]["load_checkpoint"] = True
                toml_dict["transfer"]["stage"] = "downstream"
                toml_dict["transfer"]["use_mlp_head"] = True
                with open(os.path.join(out_dir,
                                       f"{dset_id}_downstream_{sample_num}samples_tuned_mlp_head_from_supervised_pretrain_{model}_seed{seed}.toml"),
                          "w") as fh:
                    toml.dump(toml_dict, fh)

                # downstream finetune whole model with linear head from supervised pretraining:
                toml_dict = {"seed": seed,
                             "data": data_args.copy(),
                             "model": model_args.copy(),
                             "training": training_args.copy(),
                             "transfer": transfer_args.copy()}
                toml_dict["training"]["lr"] = 1e-4 / 2
                toml_dict["training"]["n_epochs"] = 200
                toml_dict["training"]["patience"] = 1e5
                toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                                                                        f"{dset_id}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                toml_dict["transfer"]["epochs_warm_up_head"] = 5
                toml_dict["transfer"]["freeze_feature_extractor"] = False
                toml_dict["transfer"]["head_lr"] = 1e-4
                toml_dict["transfer"]["layers_to_fine_tune"] = []
                toml_dict["transfer"]["load_checkpoint"] = True
                toml_dict["transfer"]["stage"] = "downstream"
                toml_dict["transfer"]["use_mlp_head"] = False
                with open(os.path.join(out_dir, f"{dset_id}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

                # downstream finetune whole model with mlp head:
                toml_dict = {"seed": seed,
                             "data": data_args.copy(),
                             "model": model_args.copy(),
                             "training": training_args.copy(),
                             "transfer": transfer_args.copy()}
                toml_dict["training"]["n_epochs"] = 200
                toml_dict["training"]["patience"] = 1e5
                toml_dict["training"]["lr"] = 1e-4 / 2
                toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                                                                        f"{dset_id}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                toml_dict["transfer"]["epochs_warm_up_head"] = 5
                toml_dict["transfer"]["freeze_feature_extractor"] = False
                toml_dict["transfer"]["head_lr"] = 1e-4
                toml_dict["transfer"]["layers_to_fine_tune"] = []
                toml_dict["transfer"]["load_checkpoint"] = True
                toml_dict["transfer"]["stage"] = "downstream"
                toml_dict["transfer"]["use_mlp_head"] = True
                with open(os.path.join(out_dir, f"{dset_id}_downstream_{sample_num}samples_mlp_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

                ################################################
                # Downstream transfering from MLM pretraining:

                # # downstream finetune linear head:
                # toml_dict = {"seed": seed,
                #              "data": data_args,
                #              "model": model_args,
                #              "training": training_supervised_downstream_args,
                #              "transfer": transfer_downstream_args}
                # toml_dict["transfer"]["load_checkpoint"] = True
                # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                #                                                         f"{dset_id}_mlm_pretraining_ft-trans_seed{seed}/checkpoint.pt")
                # toml_dict["transfer"]["freeze_feature_extractor"] = True
                # toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                # toml_dict["transfer"]["epochs_warm_up_head"] = 500
                # toml_dict["transfer"]["head_lr"] = 0.0001
                # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                # with open(os.path.join(out_dir,
                #                        f"{dset_id}_downstream_{sample_num}samples_tuned_linear_head_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
                #           "w") as fh:
                #     toml.dump(toml_dict, fh)

                # # downstream finetune mlp head from MLM pretraining:
                # toml_dict = {"seed": seed,
                #              "data": data_args.copy(),
                #              "model": model_args.copy(),
                #              "training": training_args.copy(),
                #              "transfer": transfer_args.copy()}
                # toml_dict["training"]["n_epochs"] = 200
                # toml_dict["training"]["patience"] = 1e5
                # toml_dict["training"]["lr"] = 1e-4
                # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                #                                                         f"{dset_id}_mlm_pretraining_ft-trans_seed{seed}/checkpoint.pt")
                # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                # toml_dict["transfer"]["epochs_warm_up_head"] = 0
                # toml_dict["transfer"]["freeze_feature_extractor"] = True
                # toml_dict["transfer"]["head_lr"] = 1e-4
                # toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                # toml_dict["transfer"]["load_checkpoint"] = True
                # toml_dict["transfer"]["stage"] = "downstream"
                # toml_dict["transfer"]["use_mlp_head"] = True
                # with open(os.path.join(out_dir,
                #                        f"{dset_id}_downstream_{sample_num}samples_tuned_mlp_head_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
                #           "w") as fh:
                #     toml.dump(toml_dict, fh)
                #
                # # downstream finetune whole model with linear head:
                # toml_dict = {"seed": seed,
                #              "data": data_args.copy(),
                #              "model": model_args.copy(),
                #              "training": training_args.copy(),
                #              "transfer": transfer_args.copy()}
                # toml_dict["training"]["lr"] = 1e-4 / 2
                # toml_dict["training"]["n_epochs"] = 200
                # toml_dict["training"]["patience"] = 1e5
                # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                #                                                         f"{dset_id}_mlm_pretraining_ft-trans_seed{seed}/checkpoint.pt")
                # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                # toml_dict["transfer"]["epochs_warm_up_head"] = 5
                # toml_dict["transfer"]["freeze_feature_extractor"] = False
                # toml_dict["transfer"]["head_lr"] = 1e-4
                # toml_dict["transfer"]["layers_to_fine_tune"] = []
                # toml_dict["transfer"]["load_checkpoint"] = True
                # toml_dict["transfer"]["stage"] = "downstream"
                # toml_dict["transfer"]["use_mlp_head"] = False
                # with open(os.path.join(out_dir,
                #                        f"{dset_id}_downstream_{sample_num}samples_linear_head_tuned_full_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
                #           "w") as fh:
                #     toml.dump(toml_dict, fh)
                #
                # # downstream finetune whole model with mlp head:
                # toml_dict = {"seed": seed,
                #              "data": data_args.copy(),
                #              "model": model_args.copy(),
                #              "training": training_args.copy(),
                #              "transfer": transfer_args.copy()}
                # toml_dict["training"]["n_epochs"] = 200
                # toml_dict["training"]["patience"] = 1e5
                # toml_dict["training"]["lr"] = 1e-4 / 2
                # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                #                                                         f"{dset_id}_mlm_pretraining_ft-trans_seed{seed}/checkpoint.pt")
                # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                # toml_dict["transfer"]["epochs_warm_up_head"] = 5
                # toml_dict["transfer"]["freeze_feature_extractor"] = False
                # toml_dict["transfer"]["head_lr"] = 1e-4
                # toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                # toml_dict["transfer"]["load_checkpoint"] = True
                # toml_dict["transfer"]["stage"] = "downstream"
                # toml_dict["transfer"]["use_mlp_head"] = True
                # with open(os.path.join(out_dir,
                #                        f"{dset_id}_downstream_{sample_num}samples_mlp_head_tuned_full_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
                #           "w") as fh:
                #     toml.dump(toml_dict, fh)
                supervised_pretraining_path_checkpoint = os.path.join(out_dir, f"{dset_id}_supervised_pretraining_{model}_seed{seed}", 'checkpoint.pt')
                launch_str = launch_str + f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_downstream_{sample_num}samples_fromscratch_{model}_seed{seed}.toml -f \n" \
                                          f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_downstream_{sample_num}samples_random_extractor_{model}_seed{seed}.toml -f \n" \
                                          f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_downstream_{sample_num}samples_tuned_linear_head_from_supervised_pretrain_{model}_seed{seed}.toml -f \n" \
                                          f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_downstream_{sample_num}samples_tuned_mlp_head_from_supervised_pretrain_{model}_seed{seed}.toml -f \n" \
                                          f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_downstream_{sample_num}samples_linear_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml -f \n" \
                                          f"python -W ignore bin/{model}.py output/deep_configs/{dset_id}_downstream_{sample_num}samples_mlp_head_tuned_full_from_supervised_pretrain_{model}_seed{seed}.toml -f \n"

            launch_str += f"python destructive_danger/delete_checkpoint.py --file_root {supervised_pretraining_path_checkpoint}\n"
            os.makedirs('../RTDL/launch/tuned_deep_jobs', exist_ok=True)
            with open(f"../RTDL/launch/tuned_deep_jobs/{dset_id}_{model}_exp_seed{seed}.sh", "w") as fh:
                fh.write(launch_str)