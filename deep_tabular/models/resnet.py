""" resnet.py
    ResNet model calss
    Adopted from https://github.com/Yura52/rtdl
    March 2022
"""

import math
import typing as ty
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self,
                 d_numerical,
                 d_out,
                 categories,
                 d_embedding,
                 d,
                 d_hidden_factor,
                 n_layers,
                 activation,
                 normalization,
                 hidden_dropout,
                 residual_dropout):
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = F.relu
        self.last_activation = F.relu
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

    def forward(self, x_num, x_cat):
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


def resnet(num_numerical, unique_categories, num_outputs, d_embedding, model_params):
    return ResNet(num_numerical, num_outputs, unique_categories, d_embedding,
                  model_params.d,
                  model_params.d_hidden_factor,
                  model_params.n_layers,
                  model_params.activation,
                  model_params.normalization,
                  model_params.hidden_dropout,
                  model_params.residual_dropout)
