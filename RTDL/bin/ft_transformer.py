# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import zero
from torch import Tensor
import os

import lib


# %%
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            # print('Emb shape:', self.category_embeddings.weight.shape)
            # print(self.category_offsets)
            # print(x_cat)
            # print(x_cat.shape, self.category_offsets.shape)
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

import os
import errno
import shutil
def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

# %%
if __name__ == "__main__":
    args, output = lib.load_config()
    args['model'].setdefault('token_bias', True)
    args['model'].setdefault('kv_compression', None)
    args['model'].setdefault('kv_compression_sharing', None)
    print('Args:', args)

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

    # laod data (Numerical features, Categorical features, labels, and dictionary with info for the data)
    # N, C, y, info = lib.data_prep_openml(ds_id = dset_id, seed = args['seed'], task = args['data']['task'], datasplit=[.65, .15, .2])
    #####################################################################################
    # TRANSFER#
    #####################################################################################
    N, C, y, info, full_cat_data_for_encoder = lib.data_prep_openml_transfer(ds_id=dset_id,
                                                  seed=args['seed'],
                                                  task=args['data']['task'],
                                                  stage=args['transfer']['stage'],
                                                  datasplit=[.65, .15, .2],
                                                  pretrain_proportion=args['transfer']['pretrain_proportion'],
                                                  downstream_samples_per_class=args['transfer']['downstream_samples_per_class'],
                                                  column_mode=args['transfer']['column_mode'],
                                                  pretrain_subsample=args['transfer']['pretrain_subsample'])

    # Arpit - Added column_mode for imputation
    # Arpit - This column_mode will add Information about whether to remove the column from the upstream task or the downstream task.
    # Arpit - Also will tell whether the mode is to impute the missing column.
    #####################################################################################
    # TRANSFER#
    #####################################################################################

    stats['replacement_sampling'] = info['replacement_sampling']
    if args['data']['task'] == 'multiclass':
        stats['num_classes_train'] = len(set(y['train']))
        stats['num_classes_test'] = len(set(y['test']))
    else:
        stats['num_classes_train'] = np.nan
        stats['num_classes_test'] = np.nan

    stats['num_training_samples'] = len(y['train'])
    if C is not None:
        stats['cat_features_no'] = C['train'].shape[1]
    else:
        stats['cat_features_no'] = 0
    if N is not None:
        stats['num_features_no'] = N['train'].shape[1]
    else:
        stats['num_features_no'] = 0

    D = lib.Dataset(N, C, y, info)

    X = D.build_X(
        normalization=args['data']['normalization'],
        num_nan_policy='mean',   # replace missing values in numerical features by mean
        cat_nan_policy='new',    # replace missing values in categorical features by new values
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
        full_cat_data_for_encoder = full_cat_data_for_encoder
    )

    if not isinstance(X, tuple):
        X = (X, None)
    zero.set_randomness(args['seed'])

    Y, y_info = D.build_y(args['data'].get('y_policy'))

    print('\n Y: {} {} \n'.format(Y['train'].sum(axis = 0), Y['train'].shape))
    # lib.dump_pickle(y_info, output / 'y_info.pickle')

    if args['transfer']['column_mode'] == 'train_to_predict_missing_column':

        print(y_info)

        create_folder('./predict_missing_column/')
        if args['transfer']['stage'] == 'pretrain':
            torch.save(y_info, f"./predict_missing_column/y_info_predicted_column_using_upstream_mimic_{args['seed']}_{args['transfer']['pretrain_proportion']}.pt")
        elif args['transfer']['stage'] == 'downstream':
            torch.save(y_info,
                       f"./predict_missing_column/y_info_predicted_column_using_downstream_mimic_{args['seed']}_{args['transfer']['downstream_samples_per_class']}_{args['transfer']['pretrain_proportion']}.pt")

    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)

    device = lib.get_device()
    print('Device is {}'.format(device))
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y

    X_num, X_cat, _, _ = X

    del X
    #do we think we might need to not convert to float here if binclass since we want multilabel?
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    #Constructing loss function, model and optimizer

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
    eval_batch_size = args['training']['eval_batch_size']
    print('Train size is {}, batch_size is {}, epoch_size is {}, eval_batch_size is {}'.format(train_size, batch_size,
                                                                                               epoch_size, eval_batch_size))

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    print('Loss fn is {}'.format(loss_fn))

    print('\n CATEGORIES:{}\n'.format(lib.get_categories_full_cat_data(full_cat_data_for_encoder)))
    model = Transformer(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories_full_cat_data(full_cat_data_for_encoder), # I think it's # of unique values in each cat variable
        d_out=D.info['n_classes'] if D.is_multiclass or D.is_binclass else 1, #multilabel in case of pretraining binclass
        **args['model'],
    ).to(device)

    ## Arpit - Predict the missing column for the dataset
    ## should save both for upstream and downstream
    ## args['transfer']['checkpoint_path'] should have the model which predicts the missing column

    if args['transfer']['column_mode'] == 'predict_missing_column':

        print(args['transfer']['checkpoint_path'])

        column_pred_checkpoint = torch.load(args['transfer']['checkpoint_path'])
        pretrained_feature_extractor_dict = {k: v for k, v in column_pred_checkpoint['model'].items() if
                                             'head' not in k}
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_feature_extractor_dict, strict=False)
        print('\n Loaded \n Missing keys:{}\n Unexpected keys:{}'.format(missing_keys, unexpected_keys))
        model.load_state_dict(column_pred_checkpoint['model'])
        model = model.eval()

        for part in lib.PARTS:
            predictions = None
            print(D.size(part))

            for batch_idx in lib.IndexLoader(D.size(part), 256, False, device):
                X_num_batch = None if X_num is None else X_num[part][batch_idx].float()
                X_cat_batch = None if X_cat is None else X_cat[part][batch_idx]

                model_output = model(X_num_batch, X_cat_batch).detach().cpu()
                if predictions is None:
                    predictions = model_output
                else:
                    predictions = torch.cat([predictions, model_output], dim=0)
                # since it is regression saving the output as it is.

            create_folder('./predict_missing_column/')
            y_imp = Y[part]

            print(torch.sqrt( torch.mean((y_imp - predictions)**2) ))

            # what is it predicted on is stage
            if args['transfer']['stage'] == 'pretrain':

                y_info = torch.load(f"./predict_missing_column/y_info_predicted_column_using_downstream_mimic_{args['seed']}_{args['transfer']['downstream_samples_per_class']}_{args['transfer']['pretrain_proportion']}.pt")
                predictions = predictions * y_info['std'] + y_info['mean']

                print(y_info)
                print(len(predictions))


                torch.save(predictions,
                           f"./predict_missing_column/predicted_column_using_downstream_on_upstream_{part}_mimic_{args['seed']}_{args['transfer']['downstream_samples_per_class']}_{args['transfer']['pretrain_proportion']}.pt")
            elif args['transfer']['stage'] == 'downstream':

                y_info = torch.load(f"./predict_missing_column/y_info_predicted_column_using_upstream_mimic_{args['seed']}_{args['transfer']['pretrain_proportion']}.pt")
                predictions = predictions * y_info['std'] + y_info['mean']

                print(y_info)
                print('downstream')
                print(len(predictions))

                torch.save(predictions,
                           f"./predict_missing_column/predicted_column_using_upstream_on_downstream_{part}_mimic_{args['seed']}_{args['transfer']['downstream_samples_per_class']}_{args['transfer']['pretrain_proportion']}.pt")
        exit()

    #####################################################################################
    #TRANSFER#
    #####################################################################################
    if ('downstream' in args['transfer']['stage']):
        if (args['transfer']['load_checkpoint']):
            print('Loading checkpoint, doing transfer learning')
            pretrain_checkpoint = torch.load(args['transfer']['checkpoint_path'])

            pretrained_feature_extractor_dict = {k: v for k, v in pretrain_checkpoint['model'].items() if 'head' not in k}
            missing_keys, unexpected_keys = model.load_state_dict(pretrained_feature_extractor_dict, strict = False)
            print('\n Loaded \n Missing keys:{}\n Unexpected keys:{}'.format(missing_keys, unexpected_keys))
        # except:
        #     model.load_state_dict(lib.remove_parallel(pretrain_checkpoint['model']))

        if args['transfer']['use_mlp_head']:
            emb_dim = model.head.in_features
            out_dim = model.head.out_features
            model.head = nn.Sequential(
                nn.Linear(emb_dim, 200),
                nn.ReLU(),
                nn.Linear(200, 200),
                nn.ReLU(),
                nn.Linear(200, out_dim)).to(device)

        #Freeze feature extractor
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

    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = lib.get_n_parameters(model)

    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    for x in ['tokenizer', '.norm', '.bias']:
        assert any(x in a for a in (b[0] for b in model.named_parameters()))
    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
    print('\n\n HEAD LR {}: {}\n\n'.format(args['transfer']['head_lr'], np.isnan(args['transfer']['head_lr']) ))

    ###############################################################
    # TRANSFER: differential learning rates for head and feat extr
    ###############################################################
    if ('downstream' in args['transfer']['stage']) and (not np.isnan(args['transfer']['head_lr'])):
        parameters_with_wd = [v for k, v in model.named_parameters() if (needs_wd(k)) and ('head' not in k)]
        parameters_without_wd = [v for k, v in model.named_parameters() if (not needs_wd(k)) and ('head' not in k)]
        head_parameters = [v for k, v in model.named_parameters() if 'head' in k]
        optimizer = lib.make_optimizer(
            args['training']['optimizer'],
            (
                [
                    {'params': parameters_with_wd},
                    {'params': parameters_without_wd, 'weight_decay': 0.0},
                    {'params': head_parameters, 'lr': args['transfer']['head_lr']}
                ]
            ),
            args['training']['lr'],
            args['training']['weight_decay'],
        )
    else:
        optimizer = lib.make_optimizer(
            args['training']['optimizer'],
            (
                [
                    {'params': parameters_with_wd},
                    {'params': parameters_without_wd, 'weight_decay': 0.0},
                ]
            ),
            args['training']['lr'],
            args['training']['weight_decay'],
        )
    ###############################################################
    # TRANSFER: differential learning rates for head and feat extr
    ###############################################################
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
    epoch_idx = 0
    #If doing head warmup
    if args['transfer']['epochs_warm_up_head'] > 0:
        lib.freeze_parameters(model, ['head'])
        head_warmup_flag = True
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()

        model.train()
        epoch_losses = []
        cur_batch = 0
        for batch_idx in epoch:
            ###########
            # Transfer: head warmup
            ###########
            #If doing head warmup
            if args['transfer']['epochs_warm_up_head'] > 0:
                if head_warmup_flag:
                    if epoch_idx >= args['transfer']['epochs_warm_up_head']:
                        #Stop warming up head after a predefined number of batches
                        lib.unfreeze_all_params(model)
                        head_warmup_flag = False
            ###########
            # Transfer: head warmup
            ###########

            ###########
            #Transfer: lr warmup
            ###########
            # if epoch_idx*epoch_size + cur_batch + 1 <= args['training']['num_batch_warm_up']:  # adjust LR for each training batch during warm up
            #     lib.warm_up_lr(epoch_idx*epoch_size + cur_batch + 1, args['training']['num_batch_warm_up'], args['training']['lr'], optimizer)
            ###########
            # Transfer: lr warmup
            ###########
            #random_state = zero.get_random_state()
            #zero.set_random_state(random_state)

            X_num_batch = None if X_num is None else X_num['train'][batch_idx].float()
            X_cat_batch = None if X_cat is None else X_cat['train'][batch_idx]

            optimizer.zero_grad()
            model_output = model(X_num_batch, X_cat_batch)
            # print(model_output, Y_device['train'][batch_idx], Y_device['train'][batch_idx].dtype)
            #Could use .squeeze() in here
            loss = loss_fn(model_output, Y_device['train'][batch_idx])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())
            cur_batch += 1

        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[lib.VAL]['score'])

        #Record metrics every 5 epochs on downstream tasks:
        if 'downstream' in args['transfer']['stage']:
            if epoch_idx % 1 == 0:
                stats['Epoch_{}_metrics'.format(epoch_idx)], predictions = evaluate(lib.PARTS)
                stats['Epoch_{}_metrics'.format(epoch_idx)][lib.TRAIN]['train_loss'] = sum(epoch_losses) / len(epoch_losses)

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail: # stopping criterion is based on val accuracy (see patience arg in args)
            break
        epoch_idx += 1
    # %%
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(lib.PARTS)
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
    stats['time'] = lib.format_seconds(timer())
    save_checkpoint(True)
    print('Done!')

    if 'downstream' in args['transfer']['stage']:
        if args['transfer']['column_mode'] != 'train_to_predict_missing_column':
            os.remove(checkpoint_path)
