import toml
import os


# Defining parameters
models = ['ft_transformer', 'resnet']
dset_ids = ['mimic']
downstream_targets = [0,1,4,5]
big_dsets = []
multiclass_dsets = []
regression_dsets = []
binclass_dsets = ['mimic']
sample_nums = [2, 5, 10, 50, 100]
out_dir = f"output_mimic_dev/deep_upstream_subsample_tuning"
os.makedirs(out_dir, exist_ok=True)
time = 10
global_launch_str = "\n"


def get_model_config(model, dset_big, downstream_target):

    base_config_data = {
        "cat_policy": 'indices',
        "normalization": 'quantile',
    }
    ######### FT ############################
    if model == 'ft_transformer':

        base_config_model = {
            "activation": 'reglu',
	        "initialization": 'kaiming',
	        "n_heads": 8,
	        "prenormalization": True
        }


        optimization_space_model = {
            "attention_dropout": ['uniform', 0.0, 0.5],
            "d_ffn_factor": [ '$d_ffn_factor', 1.0, 4.0 ],
            "d_token": ['$d_token', 64, 512],
            "ffn_dropout": ['uniform', 0.0, 0.5],
            "n_layers": ['int', 1, 4],
            "residual_dropout": ['?uniform', 0.0, 0.0, 0.2]
        }

        optimization_space_training = {
            "lr": ['loguniform', 1e-05, 0.001],
            "weight_decay": ['loguniform', 1e-06, 0.001],
        }

        if dset_big:
            base_config_model['d_ffn_factor'] = 4/3
            base_config_model['residual_dropout'] = 0.0
            optimization_space_model['n_layers'] = ['int', 1, 6]
            optimization_space_training['lr'] = ['loguniform', 3e-05, 3e-04]
            del optimization_space_model['d_ffn_factor']
            del optimization_space_model['residual_dropout']

    ################### SAINT #############33
    elif model == 'saint':

        base_config_model = {
            "cont_embeddings": 'MLP',
            "attentiontype": 'colrow',
            "final_mlp_style": 'sep',
            "use_cls": True,
            "depth": 1
        }

        optimization_space_model = {
            "attn_dropout": [ 'uniform', 0.2, 0.8],
            "ff_dropout": [ 'uniform', 0.2, 0.8],
            "heads": ['int', 2, 8],
            "embed_dim": ['int', 4, 32],
        }

        optimization_space_training = {
            "lr": ['loguniform', 1e-05, 0.001],
            "weight_decay": ['loguniform', 1e-06, 0.001],
        }

    if model == 'mlp':

        base_config_model = {
        }

        optimization_space_model = {
            "d_layers": [ '$mlp_d_layers', 1, 8, 1, 512 ],
            "dropout": [ '?uniform', 0.0, 0.0, 0.5],
            "d_embedding": ['int', 64, 512],
        }

        optimization_space_training = {
            "lr": ['loguniform', 1e-05, 0.01],
            "weight_decay": ['loguniform', 1e-06, 0.001],
        }

        if dset_big:
            optimization_space_model['d_layers'] = [ '$mlp_d_layers', 1, 16, 1, 1024]


    if model == 'resnet':

        base_config_model = {
            "activation": 'relu',
	        "normalization": 'batchnorm',
        }

        optimization_space_model = {
            "d": [ 'int', 64, 512 ],
            "d_embedding": ['int', 64, 512],
            "d_hidden_factor": [ 'uniform', 1.0, 4.0 ],
            "hidden_dropout": [ 'uniform', 0.0, 0.5 ],
            "n_layers": [ 'int', 1, 8 ],
            "residual_dropout": [ '?uniform', 0.0, 0.0, 0.5 ],
        }

        optimization_space_training = {
            "lr": ['loguniform', 1e-05, 0.01],
            "weight_decay": ['loguniform', 1e-06, 0.001],
        }

        if dset_big:
            optimization_space_model['d'] = [ 'int', 64, 1024]
            optimization_space_model['n_layers'] = [ 'int', 1, 16]

    if model == 'tab_transformer':

        base_config_model = {
            "dim_head": 16,
            "mlp_hidden_mults": [4, 2],
        }

        optimization_space_model = {
            "heads": ['int', 2, 8],
            "dim": ['int', 8, 128],
            "depth": ['int', 1, 12],
            "attn_dropout": [ 'uniform', 0.0, 0.5],
            "ff_dropout": [ 'uniform', 0.0, 0.5],
        }

        optimization_space_training = {
            "lr": ['loguniform', 1e-06, 0.001],
            "weight_decay": ['loguniform', 1e-06, 0.001],
        }

    base_config_training = {
        "batch_size": 256,
        "eval_batch_size": 256,
        "n_epochs": 500,
        "optimizer": 'adamw',
        "patience": 30,
        "lr_n_decays": 0,
        "num_batch_warm_up": 0
    }


    if model == 'saint':
        del base_config_training['eval_batch_size']

    base_config_transfer = {
        "stage": 'pretrain',
        "load_checkpoint": False,
        "checkpoint_path": float("nan"),
        "pretrain_proportion": downstream_target,
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

    return base_config_data, base_config_model, optimization_space_model, base_config_training, optimization_space_training, \
           base_config_transfer, optimization_options, optimization_sampler


for model in models:
    for downstream_target in downstream_targets:
        for sample_num in sample_nums:
            for dset_id in dset_ids:
                dset_is_big = dset_id in big_dsets
                base_config_data, base_config_model, optimization_space_model, base_config_training, optimization_space_training, \
                   base_config_transfer, optimization_options, optimization_sampler = get_model_config(model, dset_is_big, downstream_target)

                #upstream subsample tuning!
                base_config_transfer['pretrain_subsample'] = True
                base_config_data['dset_id'] = dset_id
                base_config_transfer['downstream_samples_per_class'] = sample_num
                if dset_id in multiclass_dsets:
                    base_config_data['task'] = 'multiclass'
                elif dset_id in regression_dsets:
                    base_config_data['task'] = 'regression'
                    base_config_data['y_policy'] = 'mean_std'
                elif dset_id in binclass_dsets:
                    base_config_data['task'] = 'binclass'
                else:
                    raise NameError('Dset_id is not known')

                if dset_id in ['aloi', 41169, 554]:
                    base_config_data["normalization"] = 'standard'

                if dset_id in ['aloi', 41166]:
                    optimization_options["n_trials"] = 30

                toml_dict = {"program": 'bin/' + model + '.py'}
                toml_dict['base_config'] = {
                    "seed": 0,
                    "data": base_config_data,
                    "model": base_config_model,
                    "training": base_config_training,
                    "transfer": base_config_transfer
                }

                toml_dict['optimization'] = {
                    "options": optimization_options,
                    "sampler": optimization_sampler,
                    "space": {"model": optimization_space_model,
                              "training": optimization_space_training}
                }


                with open(os.path.join(out_dir, f"optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.toml"), "w") as fh:
                    toml.dump(toml_dict, fh)

                launch_str = f"python -W ignore bin/tune.py {out_dir}/optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.toml -f \n"

                os.makedirs('launch_mimic_dev/deep_upstream_subsample_tuning', exist_ok=True)
                with open(f"launch_mimic_dev/deep_upstream_subsample_tuning/optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.sh", "w") as fh:
                    fh.write(launch_str)

                global_launch_str += f"python3 dispatch_single_node.py launch_mimic_dev/deep_upstream_subsample_tuning/optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model}.sh --name optuning_{dset_id}{downstream_target}_num_samples{sample_num}_{model} --time {time} \n"

os.makedirs('launch_mimic_dev/global_launch', exist_ok=True)
with open(f"launch_mimic_dev/global_launch/deep_upstream_subsample_tuning.sh", "w") as fh:
    fh.write(global_launch_str)