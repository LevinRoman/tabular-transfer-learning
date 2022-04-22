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
datasets = ['mimic'+str(i) for i in [0,1,4,5]]
models = ['catboost', 'ft_transformer', 'resnet']# 'mlp', 'tab_transformer']
catboost_setups = ['fromscratch']
deep_setups = ['fromscratch', 'linear_head_tuned_full_from_supervised_pretrain', 'mlp_head_tuned_full_from_supervised_pretrain',
               'tuned_linear_head_from_supervised_pretrain', 'tuned_mlp_head_from_supervised_pretrain']
deep_dirs = ['deep_dwnstrm_default_batchsize32_no_resampling', 'deep_dwnstrm_default_batchsize32']
catboost_dirs = ['catboost_dwnstrm_default_100','catboost_dwnstrm_default_1000', 'catboost_dwnstrm_upstream_subsample_tuned']
# deep_dirs = ['deep_dwnstrm_tuned_standard', 'deep_baselines_dwnstrm_upstream_subsample_tuned', 'deep_dwnstrm_default_standard']
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
                cur_dirs = catboost_dirs if cur_model == 'catboost' else deep_dirs
                for cur_dir in cur_dirs:
                    for cur_setup in cur_setups:
                        for cur_sample in num_samples:
                            if cur_model in ['catboost', 'xgboost']:
                                if cur_dir == 'catboost_dwnstrm_default_100':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_nestimators100' + '_seed' + str(cur_seed)
                                elif cur_dir == 'catboost_dwnstrm_default_1000':
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_nestimators1000' + '_seed' + str(cur_seed)
                                elif cur_dir == 'catboost_dwnstrm_upstream_subsample_tuned':
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
                                elif (cur_dir == 'deep_dwnstrm_default_batchsize32') or (cur_dir == 'deep_dwnstrm_default_batchsize32_no_resampling'):
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
                                            if cur_dir == 'deep_baselines_dwnstrm_upstream_subsample_tuned':
                                                with open(f"output_mimic_dev/deep_upstream_subsample_tuning/optuning_{cur_dset}_num_samples{cur_sample}_{cur_model}/stats.json", "r") as optuna_path:
                                                    upstream_subsample_tuning = json.load(optuna_path)
                                                best_upstream_subsample_epoch = upstream_subsample_tuning['best_stats']['best_epoch']
                                                roc_auc = data[f"Epoch_{cur_epoch}_metrics"]["test"]["roc_auc"]
                                                roc_auc_train = data[f"Epoch_{cur_epoch}_metrics"]["train"]["roc_auc"]
                                            else:
                                                roc_auc = data[f"Epoch_{cur_epoch}_metrics"]["test"]["roc_auc"]
                                                roc_auc_train = data[f"Epoch_{cur_epoch}_metrics"]["train"]["roc_auc"]
                                        except KeyError:
                                            print('Key error:', file_path)
                                            if 'mimic1_downstream_2samples' in file_path:
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
    parser.add_argument("--filepath", type=str, default = '/cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/mimic/tabular-transfer-learning/RTDL/output_mimic_dev')
    parser.add_argument("--force", action='store_true')
    parser.add_argument("--modelwise", action='store_true')
    args = parser.parse_args()
    experiment = 'no_resampling'
    os.makedirs('results_dev/plots/{}'.format(experiment), exist_ok=True)
    if not os.path.exists('results_dev/plots/{}/all_results.csv'.format(experiment)) or args.force:
        df = get_exp_results(args.filepath)
        df.to_csv('results_dev/plots/{}/all_results.csv'.format(experiment))
    else:
        df = pd.read_csv('results_dev/plots/{}/all_results.csv'.format(experiment))
    print(df.head())
    df_full = df.copy()
    approaches = ['deep_dwnstrm_default_batchsize32']
    plot_models = ['ft_transformer', 'resnet']
    # for cur_approach in approaches:
    if args.modelwise:
        for cur_model in plot_models:
            for cur_setup in deep_setups:
                print('Working on {}, {}...'.format(cur_model, cur_setup))
                df = df_full[(df_full.model == cur_model) & (df_full.setup == cur_setup)]
                dsets = datasets  # ['california_housing', 'year', 541, 42726, 42728, 43172, 41169, 'jannis', 1596, 40975, 1483, 40687, 188, 'aloi', 41166]
                for dset in dsets:
                    print(dset)
                    long_dataset_table = df[
                        (df["dataset"] == dset) & (df['num_samples'].isin([2, 5, 10, 50, 100, 10, 25, 50, 250, 500]))]
                    # hue_categories = np.sort(long_dataset_table['Experiment'].unique())
                    # print(hue_categories)
                    plt.figure()
                    sns.relplot(data=long_dataset_table, x='epoch', y='score', hue='dir', col='num_samples', col_wrap=2,
                                kind='line', hue_order=deep_dirs, ci=68)#style='dir'
                    os.makedirs('results_dev/plots/{}/{}_{}'.format(experiment, cur_model, cur_setup), exist_ok=True)
                    plt.savefig('results_dev/plots/{}/{}_{}/{}_auc_test.png'.format(experiment, cur_model, cur_setup, dset))
                    plt.figure()
                    sns.relplot(data=long_dataset_table, x='epoch', y='train_score', hue='dir', col='num_samples', col_wrap=2,
                                kind='line', hue_order=deep_dirs, ci=68)#style='model',
                    plt.savefig('results_dev/plots/{}/{}_{}/{}_auc_train.png'.format(experiment, cur_model, cur_setup, dset))

    else:
        approaches = ['deep_dwnstrm_default_batchsize32_no_resampling', 'deep_dwnstrm_default_batchsize32']
        for cur_approach in approaches:
            print('Working on {}...'.format(cur_approach))
            df = df_full[df_full.dir == cur_approach]
            dsets = datasets  # ['california_housing', 'year', 541, 42726, 42728, 43172, 41169, 'jannis', 1596, 40975, 1483, 40687, 188, 'aloi', 41166]
            for dset in dsets:
                print(dset)
                long_dataset_table = df[
                    (df["dataset"] == dset) & (df['num_samples'].isin([2, 5, 10, 50, 100, 10, 25, 50, 250, 500]))]
                # hue_categories = np.sort(long_dataset_table['Experiment'].unique())
                # print(hue_categories)
                plt.figure()
                sns.relplot(data=long_dataset_table, x='epoch', y='score', hue='setup', col='num_samples', col_wrap=2,
                            kind='line', style='model', hue_order=deep_setups, ci=68)
                os.makedirs('results_dev/plots/{}/{}'.format(experiment, cur_approach), exist_ok=True)
                plt.savefig('results_dev/plots/{}/{}/{}_auc_test.png'.format(experiment, cur_approach, dset))
                plt.figure()
                sns.relplot(data=long_dataset_table, x='epoch', y='train_score', hue='setup', col='num_samples',
                            col_wrap=2,
                            kind='line', style='model', hue_order=deep_setups, ci=68)
                plt.savefig('results_dev/plots/{}/{}/{}_auc_train.png'.format(experiment, cur_approach, dset))

    # df = df[df.seed == 4]
    # df = df[df["Experiment"] != "random extractor"]
    # print(tabulate(exps, headers=head, floatfmt=".2f"))

    # index = ["dataset", "model", "num_samples", "setup", 'dir']
    # table = pd.pivot_table(df, index=index, aggfunc={"score": ["mean", "sem"]})#, "Balanced Acc": values})
    # # pd.set_option('display.max_rows', None)
    # table.reset_index(inplace=True)
    # table.to_csv('results_dev/all_results_seed_averaged.csv')
    # print(table.head(100))
    # # print(table)
    #
    # summary_list = []
    # # dsets = set(table["Dataset"])
    # dsets = ['mimic'+str(i) for i in range(12)]
    # for dset in dsets:
    #     print(dset)
    #     dataset_table = table[table["dataset"].astype('str') == str(dset)]
    #     num_samples_set = set(dataset_table["num_samples"]).intersection([2, 5, 10, 50, 100, 10, 25, 50, 250, 500])
    #     for num_samples in sorted(list(num_samples_set)):
    #         print(dset, num_samples)
    #         small_table = dataset_table[dataset_table["num_samples"] == num_samples]
    #         small_table.reset_index(inplace=True)
    #         small_table.columns = [' '.join(col).strip() for col in small_table.columns.values]
    #
    #         baseline_idx = small_table["setup"].str.contains("fromscratch") | small_table["setup"].str.contains("random_extractor")
    #         baseline_rows = small_table[baseline_idx]
    #         baseline_rows.reset_index(inplace=True)
    #         best_baseline_row = baseline_rows.iloc[baseline_rows["score mean"].idxmax()]
    #         baseline_score = best_baseline_row["score mean"].item()
    #         baseline_score_sem = best_baseline_row["score sem"].item()
    #
    #         transfer_rows = small_table[~(baseline_idx)]# | small_table["Experiment"].str.contains("random"))]
    #         transfer_rows.reset_index(inplace=True)
    #         best_row = transfer_rows.iloc[transfer_rows["score mean"].idxmax()]
    #         best = best_row['score mean']
    #         best_sem = best_row['score sem']
    #         gap = best - baseline_score
    #         success = gap > 0
    #         summary_list.append([dset, num_samples, best_baseline_row['model'], best_baseline_row['dir'], best_baseline_row['setup'], baseline_score, baseline_score_sem, best, best_sem, best_row["model"], best_row['dir'], best_row['setup'], gap, success])
    #
    #
    # # print(table.round(2).to_markdown())
    # head = ["Dataset", "num_samples", "Best Baseline Model", "Baseline Dir", "Best Baseline Setup", "Baseline score", "Baseline score sem", "Best transfer score", "transfer score sem", "Best Transfer Model", "Best Dir", "Best Transfer Setup", "Gap", "Success"]
    # # print(tabulate(summary_list, headers=head, floatfmt=".2f"))
    # summary_df = pd.DataFrame(summary_list, columns = head)
    # summary_df.to_csv('results_dev/mimic_summary_with_deep_defaults.csv')
    #

if __name__ == "__main__":
    main()