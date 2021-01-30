import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=config.num_inputs, hidden_size=config.num_hidden, num_layers=config.num_layers,
                            dropout=config.num_dropout)
        self.linear = nn.Linear(in_features=config.num_hidden, out_features=config.num_outputs)

    def forward(self, X, hidden=None):
        lstm_out, hidden = self.lstm(X, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden


if __name__ == '__main__':
    from config import config
    net = LSTM(config)
    print(net)
