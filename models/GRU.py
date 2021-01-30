import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=config.num_inputs, hidden_size=config.num_hidden, num_layers=config.num_layers,
                          dropout=config.num_dropout)
        self.linear = nn.Linear(in_features=config.num_hidden, out_features=config.num_outputs)

    def forward(self, X, hidden=None):
        gru_out, hidden = self.gru(X, hidden)
        linear_out = self.linear(gru_out)
        return linear_out, hidden


if __name__ == '__main__':
    from config import config
    net = GRU(config)
    print(net)

