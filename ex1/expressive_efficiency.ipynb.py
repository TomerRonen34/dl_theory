#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import SGD
from utils import RandomStateContextManager

import bokeh.plotting as blt
from bokeh.models import Legend, LegendItem
from bokeh.io import output_notebook
output_notebook()


def count_parameters(model):
    return np.sum([np.prod(tuple(param.shape)) for param in model.parameters()])


with RandomStateContextManager(seed=34):
    n = 100000
    X = torch.linspace(-5, 5, n).reshape((n,1))
    is_bias = True

    deep_dim = 3
    deep_net = Sequential(
        Linear(1, deep_dim, bias=is_bias),
        ReLU(),
        Linear(deep_dim, deep_dim, bias=is_bias),
        ReLU(),
        Linear(deep_dim, deep_dim, bias=is_bias),
        ReLU(),
        Linear(deep_dim, 1, bias=is_bias)
    )

    num_deep_params = np.sum([np.prod(tuple(param.shape)) for param in deep_net.parameters()])
    shallow_dim = 40
    shallow_net = Sequential(
        Linear(1, shallow_dim, bias=is_bias),
        ReLU(),
        Linear(shallow_dim, 1, bias=is_bias)
    )


def plot_net_functions(deep_net, shallow_net, X):
    deep_logits = deep_net(X)
    shallow_logits = shallow_net(X)
    
    fig = blt.figure()

    legend_items = []

    deep_line = fig.line(X.flatten().data.numpy(), deep_logits.flatten().data.numpy(), color='blue')
    legend_items.append(LegendItem(
        label=f"deep net, width={deep_dim}, n_params={count_parameters(deep_net)}",
        renderers=[deep_line]
    ))

    shallow_line = fig.line(X.flatten().data.numpy(), shallow_logits.flatten().data.numpy(), color='red')
    legend_items.append(LegendItem(
        label=f"shallow net, width={shallow_dim}, n_params={count_parameters(shallow_net)}",
        renderers=[shallow_line]
    ))

    fig.add_layout(Legend(items=legend_items))

    blt.show(fig)


plot_net_functions(deep_net, shallow_net, X)


with RandomStateContextManager(seed=34):
    num_report = 10
    
    mse = MSELoss()
    optimizer = SGD(shallow_net.parameters(),
                    lr=0.005, momentum=0.5)

    epochs = 1000
    for epoch in range(epochs):
        shallow_logits = shallow_net(X)
        loss = mse(shallow_logits, deep_logits)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % int(epochs/num_report) == 0:
            print("epoch:", epoch, "loss:", loss.item())


plot_net_functions(deep_net, shallow_net, X)




