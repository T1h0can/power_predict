import torch
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, config):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(in_features=config.num_inputs, out_features=config.num_outputs)

    def forward(self, X):
        linear_out = self.linear(X)
        return linear_out


if __name__ == '__main__':
    from config import config
    net = LinearNet(config)
    print(net)
