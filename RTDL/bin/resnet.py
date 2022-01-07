# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from torch import Tensor

import lib


# %%
class ResNet(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


# %%
if __name__ == "__main__":
    args, output = lib.load_config()

    '''Building Dataset'''
    zero.set_randomness(args['seed'])
    dset_id = args['data']['dset_id']
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dset_id,
        'algorithm': Path(__file__).stem,
        **lib.load_json(output / 'stats.json'),
    }

    timer = zero.Timer()
    timer.run()

    # # laod data (Numerical features, Categorical features, labels, and dictionary with info for the data)
    # N, C, y, info = lib.data_prep_openml(ds_id = dset_id, seed = args['seed'], task = args['data']['task'], datasplit=[.65, .15, .2])
    #####################################################################################
    # TRANSFER#
    #####################################################################################
    N, C, y, info, full_cat_data_for_encoder = lib.data_prep_openml_transfer(ds_id=dset_id,
                                                                             seed=args['seed'],
                                                                             task=args['data']['task'],
                                                                             stage=args['transfer']['stage'],
                                                                             datasplit=[.65, .15, .2],
                                                                             pretrain_proportion=args['transfer'][
                                                                                 'pretrain_proportion'],
                                                                             downstream_train_data_limit=
                                                                             args['transfer'][
                                                                                 'downstream_train_data_fraction'])
    #####################################################################################
    # TRANSFER#
    #####################################################################################

    D = lib.Dataset(N, C, y, info)
    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)

    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    lib.dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat, _, _ = X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    args["model"]["d_embedding"] = args["model"].get("d_embedding", None)

    model = ResNet(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories_full_cat_data(full_cat_data_for_encoder),#lib.get_categories(X_cat),
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **args['model'],
    ).to(device)

    #####################################################################################
    # TRANSFER#
    #####################################################################################
    if ('downstream' in args['transfer']['stage']) and (args['transfer']['load_checkpoint']):
        print('Loading checkpoint, doing transfer learning')
        pretrain_checkpoint = torch.load(args['transfer']['checkpoint_path'])
        # try:
        # FAILS HERE CURRENTLY BECAUSE OF KEYERROR LOADING PRETRAINED MODEL
        # NEED DIFFERENT LOAD STATE DICT
        pretrained_feature_extractor_dict = {k: v for k, v in pretrain_checkpoint['model'].items() if 'head' not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_feature_extractor_dict, strict=False)
        print('\n Loaded \n Missing keys:{}\n Unexpected keys:{}'.format(missing_keys, unexpected_keys))
        # except:
        #     model.load_state_dict(lib.remove_parallel(pretrain_checkpoint['model']))

        # Freeze feature extractor
        if args['transfer']['freeze_feature_extractor']:
            for name, param in model.named_parameters():
                print(name, param.shape)
                if not any(x in name for x in args['transfer']['layers_to_fine_tune']):
                    # if 'head' not in name:
                    param.requires_grad = False
                else:
                    print('\n Unfrozen param {}\n'.format(name))
    else:
        print('No transfer learning')
    #####################################################################################
    # TRANSFER#
    #####################################################################################

    for name, param in model.named_parameters():
        if param.requires_grad:
            print('Trainable', name, param.shape)



    stats['n_parameters'] = lib.get_n_parameters(model)
    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        model.parameters(),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pt'

    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )

    @torch.no_grad()
    def evaluate(parts):
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            predictions[part] = torch.tensor([])
            for batch_idx in lib.IndexLoader(D.size(part), eval_batch_size, False, device):
                X_num_batch = None if X_num is None else X_num[part][batch_idx].float()
                X_cat_batch = None if X_cat is None else X_cat[part][batch_idx]

                model_output = model(X_num_batch, X_cat_batch)
                predictions[part] = torch.cat([predictions[part], model_output.cpu()])

            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part].numpy(),  # type: ignore[code]
                predictions[part].numpy(),  # type: ignore[code]
                'logits',
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
        return metrics, predictions

    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        lib.dump_stats(stats, output, final)
        lib.backup_output(output)

    # %%
    timer.run()
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:

            random_state = zero.get_random_state()
            zero.set_random_state(random_state)

            X_num_batch = None if X_num is None else X_num['train'][batch_idx].float()
            X_cat_batch = None if X_cat is None else X_cat['train'][batch_idx]

            optimizer.zero_grad()
            model_output = model(X_num_batch, X_cat_batch)
            loss = loss_fn(model_output.squeeze(), Y_device['train'][batch_idx])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())

        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[lib.VAL]['score'])

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail: # stopping criterion is based on val accuracy (see patience arg in args)
            break

    # %%
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(lib.PARTS)
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
    stats['time'] = lib.format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
