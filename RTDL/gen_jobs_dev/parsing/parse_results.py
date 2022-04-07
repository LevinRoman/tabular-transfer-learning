import argparse
import datetime
import glob
import json
import os
import tqdm
import pandas as pd
from tabulate import tabulate
import pickle

def sort_table(table):
    return sorted(table, key=lambda k: (str(k[1]), float(k[3]), float(k[4])))

seeds = list(range(5))
datasets = ['mimic'+str(i) for i in [0,1,4,5]]
models = ['catboost', 'ft_transformer', 'resnet']# 'mlp', 'tab_transformer']
catboost_setups = ['fromscratch']
deep_setups = ['fromscratch', 'linear_head_tuned_full_from_supervised_pretrain', 'mlp_head_tuned_full_from_supervised_pretrain',
               'tuned_linear_head_from_supervised_pretrain', 'tuned_mlp_head_from_supervised_pretrain']

catboost_dirs = ['catboost_dwnstrm_default_100','catboost_dwnstrm_default_1000', 'catboost_dwnstrm_upstream_subsample_tuned']
deep_dirs = ['deep_dwnstrm_default_standard', 'deep_dwnstrm_default_batchsize32']#['deep_dwnstrm_tuned_standard', 'deep_baselines_dwnstrm_upstream_subsample_tuned', 'deep_dwnstrm_default_standard', 'deep_dwnstrm_default_batchsize32']
deep_baseline_setups = ['fromscratch']
num_samples = [2, 5,10,50,100]
def get_exp_results(output_path):
    seed = []
    dataset = []
    model = []
    setup = []
    samples = []
    score = []
    paths = []
    dirs = []
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
                                elif cur_dir == 'deep_dwnstrm_default_batchsize32':
                                    if cur_sample in [2,5,10]:
                                        print(f'Skipping {cur_sample} for batch 32')
                                        continue
                                    else:
                                        file_folder = cur_dset + '_downstream_' + str(
                                            cur_sample) + 'samples_' + cur_setup + '_' + \
                                                      cur_model + '_seed' + str(cur_seed)
                                else:
                                    raise NotImplementedError(f'Wrong deep directory: {cur_dir}')

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
                                    try:
                                        if cur_dir == 'deep_baselines_dwnstrm_upstream_subsample_tuned':
                                            with open(f"output_mimic_dev/deep_upstream_subsample_tuning/optuning_{cur_dset}_num_samples{cur_sample}_{cur_model}/stats.json", "r") as optuna_path:
                                                upstream_subsample_tuning = json.load(optuna_path)
                                            best_upstream_subsample_epoch = upstream_subsample_tuning['best_stats']['best_epoch']
                                            roc_auc = data[f"Epoch_{best_upstream_subsample_epoch}_metrics"]["test"]["roc_auc"]
                                        else:
                                            roc_auc = data["Epoch_99_metrics"]["test"]["roc_auc"]
                                    except KeyError:
                                        print('Key error:', file_path)
                                        if 'mimic1_downstream_2samples' in file_path:
                                            continue
                                        else:
                                            raise KeyError()

                                score.append(roc_auc)
                                setup.append(cur_setup)
                                samples.append(cur_sample)
                                model.append(cur_model)
                                seed.append(cur_seed)
                                dataset.append(cur_dset)
                                paths.append(file_path)
                                dirs.append(cur_dir)

    experiments_df = pd.DataFrame({'dataset': dataset, 'num_samples':samples, 'model':model, 'setup': setup, 'dir': dirs, 'score': score, 'seed': seed, 'path':paths})
    return experiments_df


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--filepath", type=str, default = '/cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/mimic/tabular-transfer-learning/RTDL/output_mimic_dev')
    parser.add_argument("--force", action='store_true')
    args = parser.parse_args()
    os.makedirs('results_dev', exist_ok=True)
    if not os.path.exists('results_dev/all_results.csv') or args.force:
        df = get_exp_results(args.filepath)
        df.to_csv('results_dev/all_results.csv')
    else:
        df = pd.read_csv('results_dev/all_results.csv')
    print(df.head())

    # df = df[df.seed == 4]
    # df = df[df["Experiment"] != "random extractor"]
    # print(tabulate(exps, headers=head, floatfmt=".2f"))

    index = ["dataset", "model", "num_samples", "setup", 'dir']
    table = pd.pivot_table(df, index=index, aggfunc={"score": ["mean", "sem"]})#, "Balanced Acc": values})
    # pd.set_option('display.max_rows', None)
    table.reset_index(inplace=True)
    table.to_csv('results_dev/all_results_seed_averaged.csv')
    print(table.head(100))
    # print(table)

    summary_list = []
    # dsets = set(table["Dataset"])
    dsets = ['mimic'+str(i) for i in range(12)]
    for dset in dsets:
        print(dset)
        dataset_table = table[table["dataset"].astype('str') == str(dset)]
        num_samples_set = set(dataset_table["num_samples"]).intersection([2, 5, 10, 50, 100, 10, 25, 50, 250, 500])
        for num_samples in sorted(list(num_samples_set)):
            print(dset, num_samples)
            small_table = dataset_table[dataset_table["num_samples"] == num_samples]
            small_table.reset_index(inplace=True)
            small_table.columns = [' '.join(col).strip() for col in small_table.columns.values]

            baseline_idx = small_table["setup"].str.contains("fromscratch") | small_table["setup"].str.contains("random_extractor")
            baseline_rows = small_table[baseline_idx]
            baseline_rows.reset_index(inplace=True)
            best_baseline_row = baseline_rows.iloc[baseline_rows["score mean"].idxmax()]
            baseline_score = best_baseline_row["score mean"].item()
            baseline_score_sem = best_baseline_row["score sem"].item()

            transfer_rows = small_table[~(baseline_idx)]# | small_table["Experiment"].str.contains("random"))]
            transfer_rows.reset_index(inplace=True)
            best_row = transfer_rows.iloc[transfer_rows["score mean"].idxmax()]
            best = best_row['score mean']
            best_sem = best_row['score sem']
            gap = best - baseline_score
            success = gap > 0
            summary_list.append([dset, num_samples, best_baseline_row['model'], best_baseline_row['dir'], best_baseline_row['setup'], baseline_score, baseline_score_sem, best, best_sem, best_row["model"], best_row['dir'], best_row['setup'], gap, success])


    # print(table.round(2).to_markdown())
    head = ["Dataset", "num_samples", "Best Baseline Model", "Baseline Dir", "Best Baseline Setup", "Baseline score", "Baseline score sem", "Best transfer score", "transfer score sem", "Best Transfer Model", "Best Dir", "Best Transfer Setup", "Gap", "Success"]
    # print(tabulate(summary_list, headers=head, floatfmt=".2f"))
    summary_df = pd.DataFrame(summary_list, columns = head)
    summary_df.to_csv('results_dev/mimic_summary_with_deep_defaults.csv')


if __name__ == "__main__":
    main()