import json
import os
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import lib
import csv
import numpy as np
import matplotlib.pyplot as plt

def save_output_from_dict(out_dir, state, file_name):    # Read input information
    args = []
    values = []
    for arg, value in state.items():
        args.append(arg)
        values.append(value)
        # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fname = os.path.join(out_dir, file_name)
    fieldnames = [arg for arg in args]
    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except:
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
            # Add row for this experiment
    with open(fname, 'a') as f:
        writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
        writer.writerow({arg: value for (arg, value) in zip(args, values)})
    print('\nResults saved to '+fname+'.')


def get_accuracy(dataset, model, task, stage, data_frac, if_transfer, setup, experiment, seed, n_epochs = 200, step = 5):
    if 'pretrain' in stage:
        raise ValueError('Pretrain stage does not have epoch metrics saved')
    dir_path = os.path.join('output', dataset, model, task, stage, 'data_frac_{}'.format(data_frac), if_transfer, setup, experiment, str(seed))
    stats_path = os.path.join(dir_path, 'stats.json')
    stats = lib.load_json(stats_path)
    # print(stats['Epoch_5_metrics']['test']['balanced_accuracy'])
    balanced_accuracy = []
    for epoch in range(0, n_epochs, step):
        balanced_accuracy.append(stats['Epoch_{}_metrics'.format(epoch)]['test']['adjusted_balanced_accuracy'])
    return balanced_accuracy, stats['num_classes_test'], stats['num_classes_train'], stats['replacement_sampling']

def build_score_table(args):
    results = {}
    for dataset in args.datasets:
        dir_path = os.path.join('output', dataset)
        for model in args.models:
            for task in args.tasks:
                for experiment in args.experiment_name:
                    #Now, downstream results:
                    for data_frac in args.data_fractions:
                        results['data_frac'] = data_frac
                        #no_transfer:
                        for no_transfer_setup in args.no_transfer_setups:
                            no_transfer_train_score = []
                            no_transfer_val_score = []
                            no_transfer_test_score = []
                            # no_transfer_train_support = []
                            for seed in range(args.num_seeds):
                                no_transfer_json_path = os.path.join(dir_path, model, task, 'downstream',
                                                                  'data_frac_{}'.format(data_frac), 'no_transfer',
                                                                  no_transfer_setup, experiment, str(seed), 'stats.json')
                                no_transfer_di = lib.load_json(no_transfer_json_path)
                                if args.average:
                                    # no_transfer_train_support.append(no_transfer_di['metrics']['train']['weighted avg']['support'])
                                    no_transfer_train_score.append(no_transfer_di['metrics']['train']['score'])
                                    no_transfer_val_score.append(no_transfer_di['metrics']['val']['score'])
                                    no_transfer_test_score.append(no_transfer_di['metrics']['test']['score'])
                                else:
                                    raise NotImplementedError('saving separate seeds is tedious!')
#                                     # results['no_tr_{}_support'.format(no_transfer_setup)] = no_transfer_di['metrics']['train']['weighted avg']['support']
                                    # results['no_tr_{}_train_score'.format(no_transfer_setup)] = np.round(
                                    #     no_transfer_di['metrics']['train']['score'], 3)
                                    # results['no_tr_{}_val_score'.format(no_transfer_setup)] = np.round(
                                    #     no_transfer_di['metrics']['val']['score'], 3)
                                    # results['no_tr_{}_test_score'.format(no_transfer_setup)] = np.round(
                                    #     no_transfer_di['metrics']['test']['score'], 3)


                            if args.average:
                                # results['no_tr_{}_support'.format(no_transfer_setup)] = np.mean(no_transfer_train_support)
                                results['no_tr_{}_train_score'.format(no_transfer_setup)] = np.round(
                                    np.mean(no_transfer_train_score), 3)
                                results['no_tr_{}_val_score'.format(no_transfer_setup)] = np.round(
                                    np.mean(no_transfer_val_score), 3)
                                results['no_tr_{}_test_score'.format(no_transfer_setup)] = np.round(
                                    np.mean(no_transfer_test_score), 3)

                        #transfer
                        for transfer_setup in args.transfer_setups:
                            transfer_train_score = []
                            transfer_val_score = []
                            transfer_test_score = []
                            # transfer_train_support = []
                            for seed in range(args.num_seeds):
                                transfer_json_path = os.path.join(dir_path, model, task, 'downstream',
                                                                     'data_frac_{}'.format(data_frac), 'transfer',
                                                                     transfer_setup, experiment, str(seed),
                                                                     'stats.json')
                                transfer_di = lib.load_json(transfer_json_path)
                                if args.average:
                                    # transfer_train_support.append(transfer_di['metrics']['train']['weighted avg']['support'])
                                    transfer_train_score.append(transfer_di['metrics']['train']['score'])
                                    transfer_val_score.append(transfer_di['metrics']['val']['score'])
                                    transfer_test_score.append(transfer_di['metrics']['test']['score'])
                                else:
                                    raise NotImplementedError('saving separate seeds is tedious!')
#                                     # results['tr_{}_support'.format(transfer_setup)] = transfer_di['metrics']['train']['weighted avg']['support']
                                    # results['tr_{}_train_score'.format(transfer_setup)] = np.round(
                                    #     transfer_di['metrics']['train']['score'], 3)
                                    # results['tr_{}_val_score'.format(transfer_setup)] = np.round(
                                    #     transfer_di['metrics']['val']['score'], 3)
                                    # results['tr_{}_test_score'.format(transfer_setup)] = np.round(
                                    #     transfer_di['metrics']['test']['score'], 3)

                            if args.average:
                                # results['tr_{}_support'.format(transfer_setup)] = np.mean(
                                #     transfer_train_support)
                                results['tr_{}_train_score'.format(transfer_setup)] = np.round(
                                    np.mean(transfer_train_score), 3)
                                results['tr_{}_val_score'.format(transfer_setup)] = np.round(
                                    np.mean(transfer_val_score), 3)
                                results['tr_{}_test_score'.format(transfer_setup)] = np.round(
                                    np.mean(transfer_test_score), 3)
                        tr_test_scores = [results[key] for key in results if (key.startswith('tr_')) and ('test_score' in key)]
                        no_tr_test_scores = [results[key] for key in results if ('no_tr_' in key) and ('test_score' in key)]
                        if np.max(tr_test_scores) > np.max(no_tr_test_scores):
                            results['success'] = True
                        else:
                            results['success'] = False
                        print('Max transfer score for data_frac {}:'.format(data_frac), np.max(tr_test_scores))
                        print('Max no transfer score for data_frac {}:'.format(data_frac), np.max(no_tr_test_scores))
                        results['best_tr'] = np.max(tr_test_scores)
                        results['best_no_tr'] = np.max(no_tr_test_scores)
                        results['max_gap'] = np.max(tr_test_scores) - np.max(no_tr_test_scores)
                        save_output_from_dict(args.out_dir, results, args.file_name)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default=["188", "1596", "40664", "40685", "40687", "40975", "41166", "41169", "42734"], type=str, nargs='+')
    # parser.add_argument('--dir_path', default='output/126')
    parser.add_argument('--models', default=['ft_transformer'], type=str, nargs='+')
    parser.add_argument('--tasks', default=['multiclass_transfer'], type=str, nargs='+')
    parser.add_argument('--data_fractions', default=['2', '5', '10', '50', '250'], type=str, nargs='+')
    parser.add_argument('--no_transfer_setups', default=['original_model'], type=str, nargs='+')
    parser.add_argument('--transfer_setups', default=['head_fine_tune', 'mlp_head_fine_tune', 'full_fine_tune', 'full_mlp_head_fine_tune', 'full_fine_tune_big_lr', 'full_mlp_head_fine_tune_big_lr'], type=str, nargs='+')
    parser.add_argument('--experiments', default=['default'], type=str, nargs='+')
    args = parser.parse_args()


    for dataset in args.datasets:
        for model in args.models:
            for task in args.tasks:
                for stage in ['downstream']:
                    for experiment in args.experiments:
                        fig, ax = plt.subplots(len(args.data_fractions), 1, figsize = (25, 15))
                        for data_frac_idx, data_frac in enumerate(args.data_fractions):
                            for if_transfer in ['no_transfer', 'transfer']:
                                setups = args.no_transfer_setups if if_transfer == 'no_transfer' else args.transfer_setups
                                for setup in setups:
                                    for seed in [0]:
                                        try:
                                            balanced_accuracy, num_classes_test, num_classes_train, resampling = get_accuracy(dataset, model, task, stage, data_frac, if_transfer, setup, experiment, seed, n_epochs=200, step=5)
                                            epochs = list(range(0, 200, 5))
                                            ax[data_frac_idx].plot(epochs, balanced_accuracy, label = if_transfer+'_'+setup)
                                            ax[data_frac_idx].set_title(data_frac + 'cl_test:{} cl_train:{} resample:{}'.format(num_classes_test, num_classes_train, resampling))
                                        except:
                                            print('Failed configuration: {}_{}'.format(dataset, data_frac))
                        plt.legend()
                        fig.savefig('visualization/{}_{}_{}_{}.pdf'.format(dataset, model, task, experiment), bbox_inches = 'tight')