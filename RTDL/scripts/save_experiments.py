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



def build_score_table(args):
    results = {}
    for dataset in args.datasets:
        dir_path = os.path.join('output', dataset)
        for model in args.models:
            for task in args.tasks:
                for experiment in args.experiment_name:
                    # First, pretraining results:
                    pretrain_train_score = []
                    pretrain_val_score = []
                    pretrain_test_score = []
                    # pretrain_train_support = []
                    for seed in range(args.num_seeds):
                        #First, pretraining results:
                        pretrain_json_path = os.path.join(dir_path, model, task, 'pretrain', experiment, str(seed),
                                                 'stats.json')
                        pretrain_di = lib.load_json(pretrain_json_path)
                        if args.average:
                            # pretrain_train_support.append(pretrain_di['metrics']['train']['weighted avg']['support'])
                            pretrain_train_score.append(pretrain_di['metrics']['train']['score'])
                            pretrain_val_score.append(pretrain_di['metrics']['val']['score'])
                            pretrain_test_score.append(pretrain_di['metrics']['test']['score'])
                        else:
                            raise NotImplementedError('saving separate seeds is tedious')
                            # results = {
                            #     'dset': dataset,
                            #     'model': model,
                            #     'task': task,
                            #     'experiment': experiment,
                            #     'seed': seed,
#                             #     'pretrain_support': pretrain_di['metrics']['train']['weighted avg']['support'],
                            #     'pretrain_train_score': np.round(pretrain_di['metrics']['train']['score'], 3),
                            #     'pretrain_val_score': np.round(pretrain_di['metrics']['val']['score'], 3),
                            #     'pretrain_test_score': np.round(pretrain_di['metrics']['test']['score'], 3)
                            # }

                    if args.average:
                        results = {
                            'dset': dataset,
                            'model': model,
                            'task': task,
                            'experiment': experiment,
                            'seed': seed,
                            # 'pretrain_support': np.round(np.mean(pretrain_train_support), 3),
                            'pretrain_train_score': np.round(np.mean(pretrain_train_score), 3),
                            'pretrain_val_score': np.round(np.mean(pretrain_val_score), 3),
                            'pretrain_test_score': np.round(np.mean(pretrain_test_score), 3)
                        }

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
    parser.add_argument('--datasets', default=['1483'], type=str, nargs='+')
    # parser.add_argument('--dir_path', default='output/126')
    parser.add_argument('--models', default=['ft_transformer'], type=str, nargs='+')
    parser.add_argument('--tasks', default=['multiclass_transfer'], type=str, nargs='+')
    parser.add_argument('--data_fractions', default=['0.1', '0.01', '0.005'], type=str, nargs='+')
    parser.add_argument('--no_transfer_setups', default=['original_model'], type=str, nargs='+')
    parser.add_argument('--transfer_setups', default=['head_fine_tune', 'mlp_head_fine_tune', 'full_fine_tune', 'full_mlp_head_fine_tune'], type=str, nargs='+')
    parser.add_argument('--experiment_name', default=['default'], type=str, nargs='+')
    parser.add_argument('--num_seeds', default=1, type = int)
    parser.add_argument('--file_name', default='report.csv')
    parser.add_argument('--out_dir', default='results')
    parser.add_argument('--average', action='store_true')
    args = parser.parse_args()


    build_score_table(args)






