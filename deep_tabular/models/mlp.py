""" mlp.py
    MLP model calss
    Adopted from https://github.com/Yura52/rtdl
    March 2022
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_in, d_out, categories, d_embedding, d_layers, dropout):
        super().__init__()

        # if we have categorical data
        if categories is not None:
            # update d_in to account for the number of
            # TODO Why isn't d_in correct to begin with? Does this mean it is just dimension of numerical data?
            d_in += len(categories) * d_embedding

            # compute offsets so that categorical features do not overlap
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        self.layers = nn.ModuleList([nn.Linear(d_layers[i - 1] if i else d_in, x) for i, x in enumerate(d_layers)])
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(self.category_embeddings(x_cat + self.category_offsets[None]).view(x_cat.size(0), -1))
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


def mlp(num_numerical, unique_categories, num_outputs, d_embedding, model_params):
    return MLP(num_numerical, num_outputs, unique_categories, d_embedding, model_params.d_layers, model_params.dropout)
