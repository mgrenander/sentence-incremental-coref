import torch.nn as nn
import torch.nn.init as init
from collections import Iterable


def make_embedding(dict_size, embed_size, std=0.02):
    emb = nn.Embedding(dict_size, embed_size)
    init.normal_(emb.weight, std=std)
    return emb


def make_linear(in_features, out_features, bias=True, std=0.02):
    linear = nn.Linear(in_features, out_features, bias)
    init.normal_(linear.weight, std=std)
    if bias:
        init.zeros_(linear.bias)
    return linear


def make_ffnn(feat_size, hidden_size, output_size, dropout):
    if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
        return make_linear(feat_size, output_size)

    if not isinstance(hidden_size, Iterable):
        hidden_size = [hidden_size]
    ffnn = [make_linear(feat_size, hidden_size[0]), nn.ReLU(), dropout]
    for i in range(1, len(hidden_size)):
        ffnn += [make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), dropout]
    ffnn.append(make_linear(hidden_size[-1], output_size))
    return nn.Sequential(*ffnn)

