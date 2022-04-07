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
datasets = ['mimic'+str(i) for i in [0,1,2,3,4,5,6,7,10,11]]
models = ['catboost', 'ft_transformer']#, 'resnet', 'mlp', 'tab_transformer']
catboost_setups = ['fromscratch']
deep_setups = ['fromscratch', 'linear_head_tuned_full_from_supervised_pretrain', 'mlp_head_tuned_full_from_supervised_pretrain',
               'random_extractor', 'tuned_linear_head_from_supervised_pretrain', 'tuned_mlp_head_from_supervised_pretrain']

catboost_dirs = ['default_dwnstrm_catboost_100trees','default_dwnstrm_catboost_1000trees', 'tuned_dwnstrm_catboost_1000trees', 'tuned_dwnstrm_catboost_100trees']
deep_dirs = ['tuned_dwnstrm']
num_samples = [5,10,50,100]
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
                                    try:
                                        roc_auc = data["Epoch_99_metrics"]["test"]["roc_auc"]
                                    except KeyError:
                                        print(file_path)
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



def get_exp_results_old(filepath):
    exps = []
    #'tuned_dwnstrm',
    failed_paths = []
    damaged_files = []
    for setup_dir in ['default_dwnstrm_catboost','tuned_dwnstrm']:
                      # '/cmlscratch/avi1/post_dec_6/tab/RTDL/output/tuned_dwnstrm']:# 'tuned_dwnstrm',
        if '/' in setup_dir:
            filepath_cur = setup_dir
        else:
            filepath_cur = os.path.join(filepath, setup_dir)
        count_files = 0
        for f_name in tqdm.tqdm(glob.iglob(f"{filepath_cur}/*/stats.json", recursive=True)):
            # if "41166" not in f_name:
            #     continue
            # if setup_dir == 'tuned_dwnstrm':
            #     count_files += 1
            #     print(f_name)
            #     if count_files > 30:
            #         break
            # print(f_name)
            if datetime.datetime.fromtimestamp(os.path.getmtime(f_name)) > datetime.datetime(2022, 1, 25, 19, 0, 0, 0):
                # if "california" in f_name or "jannis" in f_name:
                with open(f_name, "r") as fp:
                    try:
                        data = json.load(fp)
                    except:
                        print('DAMAGED:', f_name)
                        damaged_files.append(f_name)

                    try:
                        if "pretraining" in f_name:
                            exp_name = " ".join(f_name.split("/")[-2].split("_")[-4:-2])
                        else:
                            exp_str = (f_name.split("/")[-2].strip(str(data["dataset"]))).strip("_")
                            exp_name = " ".join(exp_str.split("_")[2:-1])
                        if 'default' in setup_dir:
                            exp_name += '_default'
                        if 'tuned' in setup_dir:
                            exp_name += '_tuned'
                        nc = data["num_classes_test"]
                        exp_list = [exp_name,
                                    data["dataset"]+str(data['config']['transfer']['pretrain_proportion']),
                                    data["algorithm"],
                                    (data["num_training_samples"] // nc) if not pd.isna(nc) else data["num_training_samples"]//2,
                                    ]
                        if "Epoch_199_metrics" in data.keys():
                            exp_list.extend([data["Epoch_199_metrics"]["test"]["score"]])
                                             # 100 * data["Epoch_195_metrics"]["test"]["balanced_accuracy"]))
                        else:
                            if (not 'pretraining' in f_name) and (not 'catboost' in f_name) and (not 'xgboost' in f_name):
                                print(f_name)
                                failed_paths.append(f_name)
                                print('Epoch 199 is not in stats')
                                continue
                                # print(data.keys())
                                # raise ValueError('Epoch 199 is not in stats')
                            else:
                                exp_list.extend([data["metrics"]["test"]["accuracy"]])
                                             # 100 * data["metrics"]["test"]["balanced_accuracy"]))
                        exp_list.append(data["config"]["seed"])
                        exps.append(exp_list)

                    except KeyError as e:
                        # print('FAILED:', f_name, e)
                        pass
    print(failed_paths)
    print('DAMAGED:', damaged_files)
    pickle_path = 'timed_out_jobs.pkl'
    with open(pickle_path, 'wb') as handle:
        pickle.dump(failed_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return exps


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--filepath", type=str, default = '/cmlscratch/vcherepa/rilevin/tabular/tabular-transfer-learning/mimic/tabular-transfer-learning/RTDL/output')
    parser.add_argument("--force", action='store_true')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    if not os.path.exists('results/all_results.csv') or args.force:
        df = get_exp_results(args.filepath)
        df.to_csv('results/all_results.csv')
    else:
        df = pd.read_csv('results/all_results.csv')
    print(df.head())

    # df = df[df.seed == 4]
    # df = df[df["Experiment"] != "random extractor"]
    # print(tabulate(exps, headers=head, floatfmt=".2f"))

    index = ["dataset", "model", "num_samples", "setup", 'dir']
    table = pd.pivot_table(df, index=index, aggfunc={"score": ["mean", "sem"]})#, "Balanced Acc": values})
    # pd.set_option('display.max_rows', None)
    table.reset_index(inplace=True)
    table.to_csv('results/all_results_seed_averaged.csv')
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
    summary_df.to_csv('results/mimic_summary_100trees_with_defaults.csv')


if __name__ == "__main__":
    main()