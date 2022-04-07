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
catboost_setups = []
deep_setups = ['fromscratch', 'linear_head_tuned_full_from_supervised_pretrain', 'mlp_head_tuned_full_from_supervised_pretrain',
               'random_extractor', 'tuned_linear_head_from_supervised_pretrain', 'tuned_mlp_head_from_supervised_pretrain']

catboost_dirs = []
deep_dirs = ['tuned_dwnstrm']
num_samples = [5,10,50,100]
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
                                file_folder = cur_dset + '_downstream_' + str(cur_sample) + 'samples_' + cur_setup + '_' + \
                                              cur_model + '_mimic_nestimators1000' + '_seed' + str(cur_seed)
                                if '100trees' in cur_dir:
                                    file_folder = cur_dset + '_downstream_' + str(
                                        cur_sample) + 'samples_' + cur_setup + '_' + \
                                                  cur_model + '_mimic_nestimators100' + '_seed' + str(cur_seed)
                            else:
                                file_folder = cur_dset + '_downstream_' + str(cur_sample) + 'samples_' + cur_setup + '_' + \
                                              cur_model + '_seed' + str(cur_seed)

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
                                            roc_auc = data[f"Epoch_{cur_epoch}_metrics"]["test"]["roc_auc"]
                                            roc_auc_train = data[f"Epoch_{cur_epoch}_metrics"]["train"]["roc_auc"]
                                        except KeyError:
                                            print(file_path)
                                            raise KeyError()

                                        train_score.append(roc_auc_train)
                                        score.append(roc_auc)
                                        setup.append(cur_setup)
                                        samples.append(cur_sample)
                                        model.append(cur_model)
                                        seed.append(cur_seed)
                                        dataset.append(cur_dset)
                                        paths.append(file_path)
                                        dirs.append(cur_dir)
                                        epoch.append(cur_epoch)

    experiments_df = pd.DataFrame({'epoch': epoch, 'dataset': dataset, 'num_samples': samples, 'model':model, 'setup': setup, 'dir': dirs, 'score': score, 'train_score': train_score, 'seed': seed, 'path':paths})
    return experiments_df



def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--filepath", type=str, default = '/cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/mimic/tabular-transfer-learning/RTDL/output')
    parser.add_argument("--force", action='store_true')
    args = parser.parse_args()
    os.makedirs('plots', exist_ok=True)
    if not os.path.exists('plots/all_results_plot.csv') or args.force:
        df = get_exp_results(args.filepath)
        df.to_csv('plots/all_results_plot.csv')
    else:
        df = pd.read_csv('plots/all_results_plot.csv')
    print(df.head())

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
        plt.savefig('plots/{}_accuracy_test.png'.format(dset))
        plt.figure()
        sns.relplot(data=long_dataset_table, x='epoch', y='train_score', hue='setup', col='num_samples', col_wrap=2,
                    kind='line', style='model', hue_order=deep_setups, ci=68)
        plt.savefig('plots/{}_accuracy_train.png'.format(dset))



    # # df = df[df.seed == 4]
    # # df = df[df["Experiment"] != "random extractor"]
    # # print(tabulate(exps, headers=head, floatfmt=".2f"))
    #
    # index = ["dataset", "model", "num_samples", "setup", 'dir']
    # table = pd.pivot_table(df, index=index, aggfunc={"score": ["mean", "sem"]})#, "Balanced Acc": values})
    # # pd.set_option('display.max_rows', None)
    # table.reset_index(inplace=True)
    # table.to_csv('results/all_results_seed_averaged.csv')
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
    # summary_df.to_csv('results/mimic_summary_100trees_with_defaults.csv')


if __name__ == "__main__":
    main()