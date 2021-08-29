import torch
import torch.nn as nn
import torch.nn.functional as F


class Dnn(nn.Module):
    def __init__(self, input_shape, n_layers, n_nodes, bias=True, verbose=False):
        super(Dnn, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_shape, n_nodes, bias=bias)])
        for i in range(n_layers-1):
            self.layers.append(nn.Linear(n_nodes, n_nodes, bias=bias))
        self.layers.append(nn.Linear(n_nodes, 1))

        if verbose:
            print(self)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return x.view(x.size()[0])


def He_init_DNN(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)








    #
