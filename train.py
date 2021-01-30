import torch
import torch.utils.data as Data
import torch.optim as optim
from load_data import load_data_for_linear
from models.Linear import LinearNet
from utils import seed_torch, k_fold


'''
def train(net, config):
    train_ls, test_ls = [], []
    train_l_sum, test_l_sum = 0.0, 0.0
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(params=net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # for features, labels in load_data():
    features, labels = load_data()
    features = torch.tensor(features.values, dtype=torch.float32)
    labels = torch.tensor(labels.values, dtype=torch.float32)
    for i in range(config.num_fold):
        X_train, y_train, X_valid, y_valid = getKFoldData(config.num_fold, i, features, labels)
        dataset = Data.TensorDataset(X_train, y_train)
        train_iter = Data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.do_train)
        for epoch in range(config.num_epochs):
            for X, y in train_iter:
                y_pred = net(X)
                l = loss(y_pred, y.view(-1, 1))
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                # print('epoch {}/{}, loss {}'.format(epoch, config.num_epochs, l.item()))
        train_ls.append(log_rmse(net, X_train, y_train))
        if y_valid is not None:
            test_ls.append(log_rmse(net, X_valid, y_valid))
            test_l_sum += test_ls[-1]
        train_l_sum += train_ls[-1]
        print('fold %d, train rmse %f, valid rmse' % (i, train_ls[-1]))#, test_ls[-1]))
    print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (config.num_fold, train_l_sum / config.num_fold, test_l_sum / config.num_fold))
    # for epoch in range(config.num_epochs):
    #     for features, labels in load_data():
    #         for i in range(config.num_fold):
    #             X_train, y_train, X_valid, y_valid = getKFoldData(config.num_fold, i, features, labels)

            # X, y = train_features, train_labels
            # dataset = Data.TensorDataset(train_features, train_labels)
            # data_iter = Data.DataLoader(dataset, shuffle=config.do_train, batch_size=config.batch_size)
            # for X, y in data_iter:
        #     y_hat = net(X)
        #     l = loss(y_hat, y.view(-1, 1))
        #     optimizer.zero_grad()
        #     l.backward()
        #     optimizer.step()
        # print('epoch {}/{}, loss {}'.format(epoch, config.num_epochs, l.item()))
'''


def train_linear(config):
    net = LinearNet(config)
    features, labels = load_data_for_linear(config)
    k_fold(features, labels, net, config)


def train(config):
    net = LinearNet(config)
    features, labels = load_data_for_linear(config)
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, shuffle=True)
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('{}/{} epoch, loss {}'.format(epoch + 1, config.num_epochs, l.item()))


if __name__ == '__main__':
    from config import config as conf
    seed_torch(conf.random_seed)
    train(conf)
    # train_linear(conf)
    # net = LinearNet(config=conf)
    # features, labels = load_data_for_linear(conf)
    # k_fold(features, labels, net, conf)
    # for param in net.parameters():
    #     torch.nn.init.normal_(param, mean=0, std=0.01)
    # train(net, myconfig)
