import toml
import os


# Defining parameters
seeds = 5
models = ['catboost'] #ft_transformer_mlm for Avi's things
dset_ids = ['california_housing', 41169, 'jannis', 1596, 'year', 541, 40975, 1483, 42726, 42728, 40687, 188, 43172, 'aloi', 41166]
n_estimators = [10, 50, 100, 500, 1000]
out_dir = f"output/tuned_dwnstrm_catboost"
sample_nums = [2, 5, 10, 50, 100]
os.makedirs(out_dir, exist_ok=True)

for model in models:
    for dset_id in dset_ids:
        optuna_config_path = f"output/optuning_configs/boost/optuning_{dset_id}_{model}/best.toml"
        optuna_args = toml.load(optuna_config_path)

        # load data args
        data_args = optuna_args["data"]
        model_args = optuna_args["model"]
        fit_args = optuna_args["fit"]
        transfer_args = optuna_args["transfer"]



        for seed in range(seeds):
            data_args["dset_id"] = dset_id


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
                    toml_dict["model"]["iterations"] = n_est
                    toml_dict["model"]["use_best_model"] = False
                    toml_dict["model"]["early_stopping_rounds"] = None
                    toml_dict["model"]["od_pval"] = None
                    toml_dict["transfer"]["stage"] = "downstream"
                    toml_dict["transfer"]["downstream_samples_per_class"] = sample_num
                    with open(os.path.join(out_dir,
                                           f"{dset_id}_downstream_{sample_num}samples_fromscratch_{model}_nestimators{n_est}_seed{seed}.toml"),
                              "w") as fh:
                        toml.dump(toml_dict, fh)


                    launch_str += f"python -W ignore bin/{model}_.py {out_dir}/{dset_id}_downstream_{sample_num}samples_fromscratch_{model}_nestimators{n_est}_seed{seed}.toml -f \n"

            os.makedirs('launch/tuned_catboost_jobs', exist_ok=True)
            with open(f"launch/tuned_catboost_jobs/{dset_id}_{model}_exp_seed{seed}.sh", "w") as fh:
                fh.write(launch_str)