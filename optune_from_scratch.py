""" optune_from_scratch.py
    Tune neural networks using Optuna
    Developed for Tabular Transfer Learning project
    March 2022
"""

import train_net_from_scratch
import hydra
import optuna
import sys
import deep_tabular as dt
import os
import copy
from omegaconf import DictConfig, OmegaConf
import json


def sample_value_with_default(trial, name, distr, min, max, default):
    # chooses suggested or default value with 50/50 chance
    if distr == 'uniform':
        value_suggested = trial.suggest_uniform(name, min, max)
    elif distr == 'loguniform':
        value_suggested = trial.suggest_loguniform(name, min, max)
    value = value_suggested if trial.suggest_categorical(f'optional_{name}', [False, True]) else default
    return value
#

def get_parameters(model, trial):
    if model=='ft_transformer':
        model_params = {
            'd_embedding':  trial.suggest_int('d_embedding', 32, 512, step=8), #using n_heads = 8 by default
            'n_layers': trial.suggest_int('n_layers', 1, 4),
            'd_ffn_factor': trial.suggest_uniform('d_ffn_factor', 2/3, 8/3),
            'attention_dropout': trial.suggest_uniform('attention_dropout', 0.0, 0.5),
            'ffn_dropout' : trial.suggest_uniform('attention_dropout', 0.0, 0.5),
            'residual_dropout': sample_value_with_default(trial, 'residual_dropout', 'uniform', 0.0, 0.2, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform('lr', 1e-5, 1e-3),
            'weight_decay':  trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            }

    if model=='resnet':
        model_params = {
            'd_embedding':  trial.suggest_int('d_embedding', 32, 512, step=8),
            'd_hidden_factor': trial.suggest_uniform('d_hidden_factor', 1.0, 4.0),
            'n_layers': trial.suggest_int('n_layers', 1, 8,),
            'hidden_dropout': trial.suggest_uniform('residual_dropout', 0.0, 0.5),
            'residual_dropout': sample_value_with_default(trial, 'residual_dropout', 'uniform', 0.0, 0.5, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform('lr', 1e-5, 1e-3),
            'weight_decay':  sample_value_with_default(trial, 'weight_decay', 'loguniform', 1e-6, 1e-3, 0.0),
            }

    if model=='mlp':
        n_layers = trial.suggest_int('n_layers', 1, 8)
        suggest_dim = lambda name: trial.suggest_int(name, 1, 512)
        d_first = [suggest_dim('d_first')] if n_layers else []
        d_middle = ([suggest_dim('d_middle')] * (n_layers - 2) if n_layers > 2 else [])
        d_last = [suggest_dim('d_last')] if n_layers > 1 else []
        layers = d_first + d_middle + d_last

        model_params = {
            'd_embedding':  trial.suggest_int('d_embedding', 32, 512, step=8),
            'd_layers': layers,
            'dropout': sample_value_with_default(trial, 'dropout', 'uniform', 0.0, 0.5, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform('lr', 1e-5, 1e-3),
            'weight_decay':  sample_value_with_default(trial, 'weight_decay', 'loguniform', 1e-6, 1e-3, 0.0),
            }

    return model_params, training_params



def objective(trial, cfg: DictConfig, trial_configs, trial_stats):

    model_params, training_params =  get_parameters(cfg.model.name, trial) # need to suggest parameters for optuna here, probably writing a function for suggesting parameters is the optimal way

    config = copy.deepcopy(cfg) # create config for train_model with suggested parameters
    for par, value in model_params.items():
        config.model[par] = value
    for par, value in training_params.items():
        config.hyp[par] = value


    stats = train_net_from_scratch.main(config)

    trial_configs.append(config)
    trial_stats.append(stats)
    print(stats)

    return stats['val_stats']['score']


@hydra.main(config_path="config", config_name="optune_config")
def main(cfg):
    n_optuna_trials = 50

    trial_stats = []
    trial_configs = []
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    func = lambda trial: objective(trial, cfg, trial_configs, trial_stats)
    study.optimize(func, n_trials=n_optuna_trials)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    best_stats = trial_stats[best_trial.number]

    with open(os.path.join("best_stats.json"), "w") as fp:
        json.dump(best_stats, fp, indent = 4)
    with open(os.path.join("best_config.json"), "w") as fp:
        json.dump(best_trial.params, fp, indent = 4)






if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()



