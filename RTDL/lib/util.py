import argparse
import datetime
import json
import os
import pickle
import random
import shutil
import sys
import time
import typing as ty
from copy import deepcopy
from pathlib import Path
import optuna
import numpy as np
import pynvml
import pytomlpp as toml
import torch
from collections import OrderedDict

from . import env

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
PARTS = [TRAIN, VAL, TEST]

BINCLASS = 'binclass'
MULTICLASS = 'multiclass'
REGRESSION = 'regression'
TASK_TYPES = [BINCLASS, MULTICLASS, REGRESSION]


def load_json(path: ty.Union[Path, str]) -> ty.Any:
    return json.loads(Path(path).read_text())


def dump_json(x: ty.Any, path: ty.Union[Path, str], *args, **kwargs) -> None:
    Path(path).write_text(json.dumps(x, *args, **kwargs) + '\n')


def load_toml(path: ty.Union[Path, str]) -> ty.Any:
    return toml.loads(Path(path).read_text())


def dump_toml(x: ty.Any, path: ty.Union[Path, str]) -> None:
    Path(path).write_text(toml.dumps(x) + '\n')


def load_pickle(path: ty.Union[Path, str]) -> ty.Any:
    return pickle.loads(Path(path).read_bytes())


def dump_pickle(x: ty.Any, path: ty.Union[Path, str]) -> None:
    Path(path).write_bytes(pickle.dumps(x))


def load(path: ty.Union[Path, str]) -> ty.Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](path)


def load_config(
    argv: ty.Optional[ty.List[str]] = None,
) -> ty.Tuple[ty.Dict[str, ty.Any], Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE')
    parser.add_argument('-o', '--output', metavar='DIR')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('--continue', action='store_true', dest='continue_')
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)

    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if snapshot_dir and Path(snapshot_dir).joinpath('CHECKPOINTS_RESTORED').exists():
        assert args.continue_

    config_path = Path(args.config).absolute()
    output_dir = (
        Path(args.output)
        if args.output
        else config_path.parent.joinpath(config_path.stem)
    ).absolute()
    sep = '=' * (8 + max(len(str(config_path)), len(str(output_dir))))  # type: ignore[code]
    print(sep, f'Config: {config_path}', f'Output: {output_dir}', sep, sep='\n')

    assert config_path.exists()
    config = load_toml(config_path)

    if output_dir.exists():
        if args.force:
            print('Removing the existing output and creating a new one...')
            shutil.rmtree(output_dir)
            output_dir.mkdir()
        elif not args.continue_:
            backup_output(output_dir)
            print('Already done!\n')
            sys.exit()
        elif output_dir.joinpath('DONE').exists():
            backup_output(output_dir)
            print('Already DONE!\n')
            sys.exit()
        else:
            print('Continuing with the existing output...')
    else:
        print('Creating the output...')
        output_dir.mkdir()

    environment: ty.Dict[str, ty.Any] = {}
    if torch.cuda.is_available():  # type: ignore[code]
        cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
        pynvml.nvmlInit()
        environment['devices'] = {
            'CUDA_VISIBLE_DEVICES': cvd,
            'torch.version.cuda': torch.version.cuda,
            'torch.backends.cudnn.version()': torch.backends.cudnn.version(),  # type: ignore[code]
            'torch.cuda.nccl.version()': torch.cuda.nccl.version(),  # type: ignore[code]
            'driver': str(pynvml.nvmlSystemGetDriverVersion(), 'utf-8'),
        }
        if cvd:
            for i in map(int, cvd.split(',')):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                environment['devices'][i] = {
                    'name': str(pynvml.nvmlDeviceGetName(handle), 'utf-8'),
                    'total_memory': pynvml.nvmlDeviceGetMemoryInfo(handle).total,
                }

    dump_stats({'config': config, 'environment': environment}, output_dir)
    return config, output_dir


def dump_stats(stats: dict, output_dir: Path, final: bool = False) -> None:
    dump_json(stats, output_dir / 'stats.json', indent=4)
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if final:
        output_dir.joinpath('DONE').touch()
        if json_output_path:
            try:
                key = str(output_dir.relative_to(env.PROJECT_DIR))
            except ValueError:
                pass
            else:
                json_output_path = Path(json_output_path)
                try:
                    json_data = json.loads(json_output_path.read_text())
                except (FileNotFoundError, json.decoder.JSONDecodeError):
                    json_data = {}
                json_data[key] = stats
                json_output_path.write_text(json.dumps(json_data))
            shutil.copyfile(
                json_output_path,
                os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
            )


_LAST_SNAPSHOT_TIME = None


def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(env.PROJECT_DIR)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        pass
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def raise_unknown(unknown_what: str, unknown_value: ty.Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def merge_defaults(kwargs: dict, default_kwargs: dict) -> dict:
    x = deepcopy(default_kwargs)
    x.update(kwargs)
    return x


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def format_seconds(seconds: float) -> str:
    return str(datetime.timedelta(seconds=round(seconds)))


# def get_categories(
#     X_cat: ty.Optional[ty.Dict[str, torch.Tensor]]
# ) -> ty.Optional[ty.List[int]]:
#     return (
#         None
#         if X_cat is None
#         else [
#             max(len(set(X_cat[TRAIN][:, i].cpu().tolist())),
#                 len(set(X_cat[VAL][:, i].cpu().tolist())),
#                 len(set(X_cat[TEST][:, i].cpu().tolist())))
#             for i in range(X_cat[TRAIN].shape[1])
#         ]
#     )

def get_categories(
    X_cat: ty.Optional[ty.Dict[str, torch.Tensor]]
) -> ty.Optional[ty.List[int]]:
    return (
        None
        if X_cat is None
        else [
            len(set(X_cat[TRAIN][:, i].cpu().tolist()))
            for i in range(X_cat[TRAIN].shape[1])
        ]
    )

def get_categories_full_cat_data(full_cat_data_for_encoder):
    return (
        None
        if full_cat_data_for_encoder is None
        else [
            len(set(full_cat_data_for_encoder.values[:, i]))
            for i in range(full_cat_data_for_encoder.shape[1])
        ]
    )

def remove_parallel(state_dict):
    ''' state_dict: state_dict of model saved with DataParallel()
        returns state_dict without extra module level '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    return new_state_dict

def warm_up_lr(batch, num_batch_warm_up, init_lr, optimizer):
    for params in optimizer.param_groups:
        params['lr'] = batch * init_lr / num_batch_warm_up

def freeze_parameters(model, unfrozen_param_list):
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not any(x in name for x in unfrozen_param_list):
            # if 'head' not in name:
            param.requires_grad = False

def unfreeze_all_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

def get_param_distributions(model_name):
    param_distributions = {}

    if model_name == 'xgb':
        param_distributions.update(
            {
                'n_iterations': optuna.distributions.IntUniformDistribution(2, 200, step=1),
                'reg_alpha': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
                'reg_lambda': optuna.distributions.LogUniformDistribution(1e-8, 100.0),
                "subsample": optuna.distributions.UniformDistribution(0.5, 1.0),
                "learning_rate": optuna.distributions.LogUniformDistribution(1e-05, 1),
                'max_depth': optuna.distributions.IntUniformDistribution(1, 9),
                'colsample_bytree': optuna.distributions.UniformDistribution(0.5, 1.0),
                'colsample_bylevel': optuna.distributions.UniformDistribution(0.5, 1.0),
                #             "min_child_weight": optuna.distributions.LogUniformDistribution(1e-08, 1e5)
            })
    elif model_name == 'catboost':
        param_distributions.update(
            {
                'iterations': optuna.distributions.IntUniformDistribution(2, 1000, step=1),
                'bagging_temperature': optuna.distributions.UniformDistribution(0.0, 1.0),
                'depth': optuna.distributions.IntUniformDistribution(1, 9),
                'reg_lambda': optuna.distributions.LogUniformDistribution(1.0, 10.0),
                'leaf_estimation_iterations': optuna.distributions.IntUniformDistribution(1, 10),
                "learning_rate": optuna.distributions.LogUniformDistribution(1e-05, 1),
            })
    else:
        raise NotImplementedError('Specified model is not implemented')

    return param_distributions