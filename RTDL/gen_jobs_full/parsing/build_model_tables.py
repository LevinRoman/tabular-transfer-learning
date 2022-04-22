import argparse
import datetime
import glob
import json
import os
import tqdm
import pandas as pd
from tabulate import tabulate
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def sort_table(table):
    return sorted(table, key=lambda k: (str(k[1]), float(k[3]), float(k[4])))

seeds = list(range(5))
datasets = ['mimic'+str(i) for i in [0,1,2,3,4,5,6,7,8,9,10,11]]
models = ['ft_transformer', 'resnet', 'mlp', 'tab_transformer']
catboost_setups = ['fromscratch']
deep_setups = ['fromscratch', 'linear_head_tuned_full_from_supervised_pretrain', 'mlp_head_tuned_full_from_supervised_pretrain',
               'tuned_linear_head_from_supervised_pretrain', 'tuned_mlp_head_from_supervised_pretrain']

catboost_dirs = ['catboost_dwnstrm_default', 'catboost_dwnstrm_upstream_subsample_tuned']
xgboost_dirs =  ['xgboost_dwnstrm_default', 'xgboost_dwnstrm_upstream_subsample_tuned']
deep_dirs = ['deep_dwnstrm_tuned_standard', 'deep_baselines_dwnstrm_upstream_subsample_tuned']#, 'deep_dwnstrm_default_standard']
deep_baseline_setups = ['fromscratch']
num_samples = [2, 5,10,50,100]
epochs = list(range(200))
def get_exp_results(output_path):
    seed = []
    dataset = []
    model = []
    setup = []
    samples = []
    score = []
    paths = []
    dirs = []
    epoch = []
    train_score = []
    failed_paths = []
    damaged_files = []
    for cur_dset in datasets:
        for cur_seed in seeds:
            for cur_model in models:
                cur_setups = catboost_setups if cur_model in ['catboost', 'xgboost'] else deep_setups
                if cur_model == 'catboost':
                    cur_dirs = catboost_dirs
                elif cur_model == 'xgboost':
                    cur_dirs = xgboost_dirs
                else:
                    cur_dirs = deep_dirs
                for cur_dir in cur_dirs:
                    for cur_setup in cur_setups:
                        for cur_sample in num_samples:
                            if cur_model in ['catboost', 'xgboost']:
                                if cur_dir == 'xgboost_dwnstrm_default':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_nestimators100' + '_seed' + str(cur_seed)
                                elif cur_dir == 'catboost_dwnstrm_default_100':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_nestimators100' + '_seed' + str(cur_seed)
                                elif cur_dir == 'catboost_dwnstrm_default':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_nestimators1000' + '_seed' + str(cur_seed)
                                elif cur_dir == 'catboost_dwnstrm_upstream_subsample_tuned':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_upstream_subsample_tuned' + '_seed' + str(cur_seed)
                                elif cur_dir == 'xgboost_dwnstrm_upstream_subsample_tuned':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_upstream_subsample_tuned' + '_seed' + str(cur_seed)
                                else:
                                    raise NotImplementedError(f'Wrong catboost directory: {cur_dir}')
                            else:
                                if cur_dir == 'deep_dwnstrm_tuned_standard':
                                    file_folder = cur_dset + '_downstream_' + str(cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_seed' + str(cur_seed)
                                elif cur_dir == 'deep_baselines_dwnstrm_upstream_subsample_tuned':
                                    if cur_setup not in deep_baseline_setups:
                                        print(f'Skipping {cur_setup} for deep baselines')
                                        continue
                                    else:
                                        file_folder = cur_dset + '_downstream_' + str(
                                            cur_sample) + 'samples_' + cur_setup + '_' + \
                                                      cur_model + '_seed' + str(cur_seed)
                                elif cur_dir == 'deep_dwnstrm_default_standard':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_seed' + str(cur_seed)
                                elif cur_dir == 'deep_dwnstrm_default_batchsize32':
                                    if cur_sample in [2,5,10]:
                                        print(f'Skipping {cur_sample} for batch 32')
                                        continue
                                    else:
                                        file_folder = cur_dset + '_downstream_' + str(
                                            cur_sample) + 'samples_' + cur_setup + '_' + \
                                                      cur_model + '_seed' + str(cur_seed)
                                else:
                                    raise NotImplementedError(f'Wrong deep directory')

                            file_path = os.path.join(output_path, cur_dir, file_folder, 'stats.json')
                            with open(file_path, "r") as fp:
                                try:
                                    data = json.load(fp)
                                except:
                                    print('NO FILE:', file_path)
                                    raise ValueError('No file')

                                if cur_model in ['catboost', 'xgboost']:
                                    try:
                                        roc_auc = data["metrics"]["test"]["roc_auc"]
                                    except KeyError:
                                        print(file_path)
                                        raise KeyError()
                                else:
                                    for cur_epoch in epochs:
                                        try:
                                            #TODO pick the best upstream epoch for baselines
                                            if cur_dir == 'deep_baselines_dwnstrm_upstream_subsample_tuned':
                                                with open(f"output_mimic_full/deep_upstream_subsample_tuning/optuning_{cur_dset}_num_samples{cur_sample}_{cur_model}/stats.json", "r") as optuna_path:
                                                    upstream_subsample_tuning = json.load(optuna_path)
                                                best_upstream_subsample_epoch = upstream_subsample_tuning['best_stats']['best_epoch']
                                                roc_auc = data[f"Epoch_{cur_epoch}_metrics"]["test"]["roc_auc"]
                                                roc_auc_train = data[f"Epoch_{cur_epoch}_metrics"]["train"]["roc_auc"]
                                            else:
                                                roc_auc = data[f"Epoch_{cur_epoch}_metrics"]["test"]["roc_auc"]
                                                roc_auc_train = data[f"Epoch_{cur_epoch}_metrics"]["train"]["roc_auc"]
                                        except KeyError:
                                            print('Key error:', file_path)
                                            if ('mimic1_downstream_2samples' in file_path) or ('mimic2_downstream_2samples' in file_path):
                                                continue
                                            else:
                                                raise KeyError()

                                        epoch.append(cur_epoch)
                                        train_score.append(roc_auc_train)
                                        score.append(roc_auc)
                                        setup.append(cur_setup)
                                        samples.append(cur_sample)
                                        model.append(cur_model)
                                        seed.append(cur_seed)
                                        dataset.append(cur_dset)
                                        paths.append(file_path)
                                        dirs.append(cur_dir)

    experiments_df = pd.DataFrame(
        {'epoch': epoch, 'dataset': dataset, 'num_samples': samples, 'model': model, 'setup': setup, 'dir': dirs,
         'score': score, 'train_score': train_score, 'seed': seed, 'path': paths})
    return experiments_df


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--filepath", type=str, default = '/cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/mimic/tabular-transfer-learning/RTDL/output_mimic_full')
    parser.add_argument("--force", action='store_true')
    args = parser.parse_args()
    os.makedirs('results_full/model_tables', exist_ok=True)
    if not os.path.exists('results_full/model_tables/all_results.csv') or args.force:
        df = get_exp_results(args.filepath)
        df.to_csv('results_full/model_tables/all_results.csv', index = False)
    else:
        df = pd.read_csv('results_full/model_tables/all_results.csv')
    print(df.head())

    if not os.path.exists('results_full/model_tables/all_results_seed_averaged_epoch_max.csv') or args.force:
        # average over seeds
        index = ["epoch", "dataset", "model", "num_samples", "setup", 'dir']
        table = pd.pivot_table(df, index=index, aggfunc={"score": ["mean", "sem"]})  # , "Balanced Acc": values})
        table.columns = ['_'.join(col).strip() for col in table.columns.values]
        table.reset_index(inplace=True)

        print(table.head())
        # max over epochs
        # index = ["dataset", "model", "num_samples", "setup", 'dir']
        # table = pd.pivot_table(table, index = index, aggfunc={'score_mean': ['max']})
        table = table.loc[table.groupby(by = ["dataset", "model", "num_samples", "setup", 'dir'])['score_mean'].idxmax()]
        # table.columns = ['_'.join(col).strip() for col in table.columns.values]
        table.reset_index(inplace=True)
        print(table.head())

        table.to_csv('results_full/model_tables/all_results_seed_averaged_epoch_max.csv', index = False)
    else:
        table = pd.read_csv('results_full/model_tables/all_results_seed_averaged_epoch_max.csv')
    print(table.head(100))

    models_to_build = ['ft_transformer', 'resnet', 'mlp', 'tab_transformer']
    for cur_model in models_to_build:
        print('Working on {}...'.format(cur_model))
        df = table[table.model == cur_model]
        df_score = df.pivot(index=['dir', 'setup'], columns=['dataset', 'num_samples'], values='score_mean')
        print(df_score)
        df_sem = df.pivot(index=['dir', 'setup'], columns=['dataset', 'num_samples'], values='score_sem')
        os.makedirs('results_full/model_tables/{}'.format(cur_model), exist_ok = True)
        df_score.to_csv('results_full/model_tables/{}/df_score.csv'.format(cur_model), float_format = '%.4f')
        df_sem.to_csv('results_full/model_tables/{}/df_sem.csv'.format(cur_model), float_format = "%.4f")


if __name__ == "__main__":
    main()