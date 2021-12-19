import math
import typing as ty
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from torch import einsum
import numpy as np
from einops import rearrange
import lib



def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ff_encodings(x,B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout,style='col'):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed =  nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, mask = None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)

        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        else:
             for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n)
        return x


# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


#mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# main class

class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))


    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred



class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories, # Tuple with # unique values for each feature
        num_continuous, # Num of continuous features
        dim, # Embedding dim
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 0,
        attn_dropout = 0.,
        ff_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col',
        final_mlp_style = 'common',
        y_dim = 2,
        use_cls = True,
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        if use_cls:
            # insert column for cls token
            categories.insert(0, 1)
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        # for each column outpus # of unique tokens in all prevuous columns (how much we want to offset values)
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]

        # why do we want to register it?
        self.register_buffer('categories_offset', categories_offset)

        # Batch Normalization Layer for continuous features
        self.norm = nn.LayerNorm(num_continuous)
        # initializing parameters
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style
        self.use_cls = use_cls

        # If encoding continuous features using MLP, define simple_MLP as a set of MLP models for each feature or single MLP
        # for all features
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        # MLP for the last layer (?)
        self.mlp = MLP(all_dimensions, act = mlp_act)
        # Embedding layer for all features (unique cat tokens + special tokens)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        # compute categorical and continuous offsets for nan masks (lists [0,2,4,6,...])
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0)
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0)
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        # registering (?)
        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        # define embedding layers for missing/non-missing values for cont and cat features
        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)

        # something weird is here
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)

        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim,(self.total_tokens)*2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim,self.num_categories,categories)
            self.mlp2 = sep_MLP(dim,self.num_continuous,np.ones(self.num_continuous).astype(int))


        self.mlpfory = simple_MLP([dim ,1000, y_dim])
        self.pt_mlp = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])
        self.pt_mlp2 = simple_MLP([dim*(self.num_continuous+self.num_categories) ,6*dim*(self.num_continuous+self.num_categories)//5, dim*(self.num_continuous+self.num_categories)//2])

    def embed_data_mask(self, x_categ, x_cont, cat_mask, con_mask):
        '''

        :param x_categ: torch.tensor with categorical features, the first column is zeros and corresponds to cls token
        :param x_cont:  torch.tensor with continuous features
        :param cat_mask: torch.tensor with zeros at NaN positions, first column is ones and corresponds to cls token
        :param con_mask: torch.tensor with zeros at NaN positions
        :param model:
        :return:
            X_categ: torch tensor with categorical features with offset
            X_categ_enc: (batch size, # of cat features + one column for cls, embedding_dim)
            X_categ_enc: (batch size, # of cont features, embedding_dim)

        '''
        device = x_cont.device
        # moving each value of each column by sum of unique counts in prev columns (since we use the same embedding layer for
        #  all cat features, their values should not intersect)

        x_categ = x_categ + self.categories_offset.type_as(x_categ)
        # embedding layer

        x_categ_enc = self.embeds(x_categ)
        n1,n2 = x_cont.shape
        _, n3 = x_categ.shape
        # encoding continuous variables, for each cont variable use its own MLP
        if self.cont_embeddings == 'MLP':
            x_cont_enc = torch.empty(n1,n2, self.dim)
            for i in range(self.num_continuous):
                x_cont_enc[:,i,:] = self.simple_MLP[i](x_cont[:,i])
        else:
            raise Exception('This case should not work!')


        x_cont_enc = x_cont_enc.to(device)
        # offset cat_mask, each column by 2 (since we need different embeddings for missing values in different columns)
        cat_mask_temp = cat_mask + self.cat_mask_offset.type_as(cat_mask)
        con_mask_temp = con_mask + self.con_mask_offset.type_as(con_mask)

        # embed missing values as well
        cat_mask_temp = self.mask_embeds_cat(cat_mask_temp)
        con_mask_temp = self.mask_embeds_cont(con_mask_temp)
        # replace embeddings for missing values with the right embeddings
        x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
        x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]

        return x_categ, x_categ_enc, x_cont_enc


    def forward(self, x_categ, x_cont, cat_mask, con_mask):


        if self.use_cls:
            x_categ = torch.cat((torch.zeros(x_categ.shape[0],1).to(x_categ.get_device()), x_categ), dim = 1).long()
            cat_mask = torch.cat((torch.ones(cat_mask.shape[0],1).to(cat_mask.get_device()), cat_mask), dim = 1).long()

        _ , x_categ_enc, x_cont_enc = self.embed_data_mask(x_categ, x_cont, cat_mask, con_mask)
        reps = self.transformer(x_categ_enc, x_cont_enc)
        y_reps = reps[:,0,:]

        y_outs = self.mlpfory(y_reps)

        return y_outs

# %%
if __name__ == "__main__":
    args, output = lib.load_config()
    args['model'].setdefault('token_bias', True)
    args['model'].setdefault('kv_compression', None)
    args['model'].setdefault('kv_compression_sharing', None)
    print(args)

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
    N, C, y, info = lib.data_prep_openml(ds_id = dset_id, seed = args['seed'], task = args['data']['task'], datasplit=[.65, .15, .2])
    D = lib.Dataset(N, C, y, info)

    X = D.build_X(
        normalization=args['data']['normalization'],
        num_nan_policy='mean',   # replace missing values in numerical features by mean
        cat_nan_policy='new',    # replace missing values in categorical features by new values
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

    X_num, X_cat, num_nan_masks, cat_nan_masks = X


    if X_num is None:
        # this is hardcoded for saint since it needs numerical features and nan mask as input even when there are no
        # numerical features in the data
        X_num = {'train': torch.empty(X_cat['train'].shape[0], 0).long().to(device),
                 'val': torch.empty(X_cat['val'].shape[0], 0).long().to(device),
                 'test': torch.empty(X_cat['test'].shape[0], 0).long().to(device)}
        num_nan_masks = {'train': torch.empty(X_cat['train'].shape[0], 0).long().to(device),
                 'val': torch.empty(X_cat['val'].shape[0], 0).long().to(device),
                 'test': torch.empty(X_cat['test'].shape[0], 0).long().to(device)}

    del X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    '''Constructing loss function, model and optimizer'''

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

    model = SAINT(
        categories = lib.get_categories(X_cat),
        num_continuous = X_num['train'].shape[1],
        dim = args['model']['embed_dim'],
        dim_out = 1,
        depth = args['model']['depth'],
        heads = args['model']['heads'],
        attn_dropout = args['model']['attn_dropout'],
        ff_dropout = args['model']['ff_dropout'],
        mlp_hidden_mults = (4, 2),
        cont_embeddings = args['model']['cont_embeddings'],
        attentiontype = args['model']['attentiontype'],
        final_mlp_style = args['model']['final_mlp_style'],
        y_dim = D.info['n_classes'] if D.is_multiclass else 1,
        use_cls = args['model']['use_cls']
    ).to(device)


    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    stats['n_parameters'] = lib.get_n_parameters(model)

    def needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

    # TODO: need to change this if we want to not apply weight decay to some groups of parameters like ft_transformer

    parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
    parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
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
                X_num_batch = X_num[part][batch_idx].float()
                X_cat_batch = X_cat[part][batch_idx]
                X_num_mask_batch = num_nan_masks[part][batch_idx]
                X_cat_mask_batch = cat_nan_masks[part][batch_idx]

                model_output = model(X_cat_batch, X_num_batch, X_cat_mask_batch, X_num_mask_batch)
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

            X_num_batch = torch.empty(len(batch_idx), 0) if X_num is None else X_num['train'][batch_idx].float()
            X_cat_batch =  torch.empty(len(batch_idx), 0) if X_cat is None else X_cat['train'][batch_idx]
            X_num_mask_batch =  torch.empty(len(batch_idx), 0) if X_num is None else num_nan_masks['train'][batch_idx]
            X_cat_mask_batch =  torch.empty(len(batch_idx), 0) if X_cat is None else cat_nan_masks['train'][batch_idx]

            optimizer.zero_grad()
            model_output = model(X_cat_batch, X_num_batch, X_cat_mask_batch, X_num_mask_batch)
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
