import toml
import os


# Defining parameters
seeds = 5
models = ['catboost'] #ft_transformer_mlm for Avi's things
dset_ids = [188, 541, 1483, 40687, 40975, 41169, 42726, 42728, "california_housing", "jannis"]
n_estimators = [1000]
out_dir = f"../RTDL/output/catboost_configs_default"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)

for model in models:
    for dset_id in dset_ids:
        optuna_config_path = f"../RTDL/optuna_configs_pretrained/{dset_id}/{model}/0/best.toml"
        optuna_args = toml.load(optuna_config_path)

        # Use default model args
        data_args = optuna_args["data"]
        model_args = {'task_type': 'GPU'}#optuna_args["model"]
        fit_args = optuna_args["fit"]
        transfer_args = optuna_args["transfer"]

        # # args templates
        # data_args = {
        #     "task": "multiclass",
        #     "cat_policy": "indices",
        #     "normalization": "quantile",
        # }
        # model_args = {
        #     "activation": "reglu",
        #     "attention_dropout": 0.2,
        #     "d_ffn_factor": 1.333333333333333,
        #     "d_token": 192,
        #     "ffn_dropout": 0.1,
        #     "initialization": "kaiming",
        #     "n_heads": 8,
        #     "n_layers": 3,
        #     "prenormalization": True,
        #     "residual_dropout": 0.0,
        # }
        # transfer_pretraining_args = {
        #     "stage": "pretrain",
        #     "load_checkpoint": False,
        #     "checkpoint_path": float("nan"),
        #     "pretrain_proportion": 0.5,
        #     "downstream_samples_per_class": float("nan"),
        #     "freeze_feature_extractor": False,
        #     "layers_to_fine_tune": [],
        #     "use_mlp_head": False,
        #     "epochs_warm_up_head": 0,
        #     "head_lr": float("nan"),
        # }
        # training_mlm_pretraining_args = {
        #     "batch_size": 512,
        #     "eval_batch_size": 8192,
        #     "lr": 0.001,
        #     "lr_n_decays": 0,
        #     "n_epochs": 1000,
        #     "optimizer": "adamw",
        #     "patience": 100000000.0,
        #     "weight_decay": 1e-5,
        #     "self_supervised": True,
        #     "mlm_masking_proportion": 0.15,
        #     "mlm_loss_masked_only": True,
        #     "mlm_mode": "mlm supervised",
        #     "num_batch_warm_up": 0,
        #     "sf_loss_lam": 0.001,
        # }
        # training_supervised_args = {
        #     "batch_size": 256,
        #     "eval_batch_size": 8192,
        #     "lr": 0.0001,
        #     "lr_n_decays": 0,
        #     "n_epochs": 500,
        #     "optimizer": "adamw",
        #     "patience": 30,
        #     "weight_decay": 1e-5,
        #     "num_batch_warm_up": 0,
        #     "self_supervised": False,
        # }
        # training_supervised_downstream_args = {
        #     "batch_size": 128,
        #     "eval_batch_size": 8192,
        #     "lr": 0.00001,
        #     "lr_n_decays": 0,
        #     "n_epochs": 200,
        #     "optimizer": "adamw",
        #     "patience": 1000,
        #     "weight_decay": 1e-5,
        #     "num_batch_warm_up": 0,
        #     "self_supervised": False,
        # }
        # transfer_downstream_args = {
        #     "stage": "downstream",
        #     "load_checkpoint": True{model  #     "checkpoint_path": None,
        #     "pretrain_proportion": 0.5,
        #     "downstream_samples_per_class": 2,
        #     "freeze_feature_extractor": False,
        #     "layers_to_fine_tune": [],
        #     "use_mlp_head": False,
        #     "epochs_warm_up_head": 0,
        #     "head_lr": float("nan"),
        # }

        for seed in range(seeds):
            data_args["dset_id"] = dset_id

            ################################################
            # Upstream pretraining:

            # # standard supervised pretraining:
            # toml_dict = {"seed": seed,
            #              "data": data_args.copy(),
            #              "model": model_args.copy(),
            #              "fit": fit_args.copy(),
            #              "transfer": transfer_args.copy()}
            # with open(os.path.join(out_dir, f"{dset_id}_supervised_pretraining_{model}_seed{seed}.toml"), "w") as fh:
            #     toml.dump(toml_dict, fh)

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

            # launch_str = f"python -W ignore bin/{model}.py output/catboost_configs_default/{dset_id}_supervised_pretraining_ft-trans_seed{seed}.toml -f \n" \
            #              f"python -W ignore bin/{model}.py output/catboost_configs_default/{dset_id}_mlm_pretraining_ft-trans_seed{seed}.toml -f \n"
            # launch_str = f"python -W ignore bin/{model}.py output/catboost_configs_default/{dset_id}_supervised_pretraining_{model}_seed{seed}.toml -f \n"

            launch_str = "\n"
            for sample_num in sample_nums:
                for n_est in n_estimators:
                    ################################################
                    # Downstream baselines:

                    # downstream supervised learning:
                    toml_dict = {"seed": seed,
                                 "data": data_args.copy(),
                                 "model": model_args.copy(),
                                 "fit": fit_args.copy(),
                                 "transfer": transfer_args.copy()}
                    # toml_dict["model"]["iterations"] = n_est
                    toml_dict["model"]["use_best_model"] = False
                    toml_dict["model"]["early_stopping_rounds"] = None
                    toml_dict["model"]["od_pval"] = None
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    with open(os.path.join(out_dir,
                                           f"{dset_id}_downstream_{sample_num}samples_fromscratch_{model}_nestimators{n_est}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)

                    # downstream random feature extractor learning:
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
                    # toml_dict["transfer"]["freeze_feature_extractor"] = True
                    # toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                    # with open(os.path.join(out_dir,
                    #                        f"{dset_id}_downstream_{sample_num}samples_random_extractor_{model}_seed{seed}.toml"),
                    #           "w") as fh:
                    #     toml.dump(toml_dict, fh)

                    ################################################
                    # Downstream transfering from supervised pretraining:

                    # # downstream finetune linear head:
                    # toml_dict = {"seed": seed,
                    #              "data": data_args,
                    #              "model": model_args,
                    #              "training": training_supervised_downstream_args,
                    #              "transfer": transfer_downstream_args}
                    # toml_dict["transfer"]["load_checkpoint"] = True
                    # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                    #                                                         f"{dset_id}_supervised_pretraining_ft-trans_seed{seed}/checkpoint.pt")
                    # toml_dict["transfer"]["freeze_feature_extractor"] = True
                    # toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                    # toml_dict["transfer"]["epochs_warm_up_head"] = 500
                    # toml_dict["transfer"]["head_lr"] = 0.0001
                    # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    # with open(os.path.join(out_dir,
                    #                        f"{dset_id}_downstream_{sample_num}samples_default_linear_head_from_supervised_pretrain_ft-trans_seed{seed}.toml"),
                    #           "w") as fh:
                    #     toml.dump(toml_dict, fh)

                    # # downstream finetune mlp head from supervised pretraining:
                    # toml_dict = {"seed": seed,
                    #              "data": data_args.copy(),
                    #              "model": model_args.copy(),
                    #              "training": training_args.copy(),
                    #              "transfer": transfer_args.copy()}
                    # toml_dict["training"]["n_epochs"] = 200
                    # toml_dict["training"]["patience"] = 1e5
                    # toml_dict["training"]["lr"] = 1e-4
                    # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                    #                                                         f"{dset_id}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    # toml_dict["transfer"]["epochs_warm_up_head"] = 0
                    # toml_dict["transfer"]["freeze_feature_extractor"] = True
                    # toml_dict["transfer"]["head_lr"] = 1e-4
                    # toml_dict["transfer"]["layers_to_fine_tune"] = ["head"]
                    # toml_dict["transfer"]["load_checkpoint"] = True
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["use_mlp_head"] = True
                    # with open(os.path.join(out_dir,
                    #                        f"{dset_id}_downstream_{sample_num}samples_default_mlp_head_from_supervised_pretrain_{model}_seed{seed}.toml"),
                    #           "w") as fh:
                    #     toml.dump(toml_dict, fh)

                    # # downstream finetune whole model with linear head from supervised pretraining:
                    # toml_dict = {"seed": seed,
                    #              "data": data_args.copy(),
                    #              "model": model_args.copy(),
                    #              "training": training_args.copy(),
                    #              "transfer": transfer_args.copy()}
                    # toml_dict["training"]["lr"] = 1e-4 / 2
                    # toml_dict["training"]["n_epochs"] = 200
                    # toml_dict["training"]["patience"] = 1e5
                    # toml_dict["transfer"]["checkpoint_path"] = os.path.join(out_dir,
                    #                                                         f"{dset_id}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    # toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    # toml_dict["transfer"]["freeze_feature_extractor"] = False
                    # toml_dict["transfer"]["head_lr"] = 1e-4
                    # toml_dict["transfer"]["layers_to_fine_tune"] = []
                    # toml_dict["transfer"]["load_checkpoint"] = True
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["use_mlp_head"] = False
                    # with open(os.path.join(out_dir,
                    #                        f"{dset_id}_downstream_{sample_num}samples_linear_head_default_full_from_supervised_pretrain_{model}_seed{seed}.toml"),
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
                    #                                                         f"{dset_id}_supervised_pretraining_{model}_seed{seed}/checkpoint.pt")
                    # toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    # toml_dict["transfer"]["epochs_warm_up_head"] = 5
                    # toml_dict["transfer"]["freeze_feature_extractor"] = False
                    # toml_dict["transfer"]["head_lr"] = 1e-4
                    # toml_dict["transfer"]["layers_to_fine_tune"] = []
                    # toml_dict["transfer"]["load_checkpoint"] = True
                    # toml_dict["transfer"]["stage"] = "downstream"
                    # toml_dict["transfer"]["use_mlp_head"] = True
                    # with open(os.path.join(out_dir,
                    #                        f"{dset_id}_downstream_{sample_num}samples_mlp_head_default_full_from_supervised_pretrain_{model}_seed{seed}.toml"),
                    #           "w") as fh:
                    #     toml.dump(toml_dict, fh)

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
                    #                        f"{dset_id}_downstream_{sample_num}samples_default_linear_head_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
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
                    #                        f"{dset_id}_downstream_{sample_num}samples_default_mlp_head_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
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
                    #                        f"{dset_id}_downstream_{sample_num}samples_linear_head_default_full_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
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
                    #                        f"{dset_id}_downstream_{sample_num}samples_mlp_head_default_full_from_mlm_pretrain_ft-trans_seed{seed}.toml"),
                    #           "w") as fh:
                    #     toml.dump(toml_dict, fh)

                    launch_str += f"python -W ignore bin/{model}_.py output/catboost_configs_default/{dset_id}_downstream_{sample_num}samples_fromscratch_{model}_nestimators{n_est}_seed{seed}.toml -f \n"

            os.makedirs('../RTDL/launch/default_catboost_jobs', exist_ok=True)
            with open(f"../RTDL/launch/default_catboost_jobs/{dset_id}_{model}_exp_seed{seed}.sh", "w") as fh:
                fh.write(launch_str)