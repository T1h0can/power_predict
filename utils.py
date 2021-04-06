import os
import random
import numpy as np
import torch
import torch.utils.data as Data
from models.Linear import LinearNet


def getKFoldData(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k

    X_train, y_train, X_valid, y_valid = None, None, None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)  # dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def log_rmse(net, features, labels):
    loss = torch.nn.MSELoss()
    clipped_preds = torch.max(net(features), torch.tensor(1.0))
    # print(clipped_preds)
    rmse = torch.sqrt(loss(clipped_preds.log(), labels.view(-1, 1).log()))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, config):
    train_ls, test_ls = [], []
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, shuffle=True, batch_size=config.batch_size)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if (epoch + 1) % 10 == 0:
            print('epoch {}/{}, loss {}'.format(epoch + 1, config.num_epochs, l.item()))
    if test_labels is not None:
        test_ls.append(log_rmse(net, test_features, test_labels))
    # print(test_ls)
    return train_ls, test_ls


def k_fold(X_train, y_train, net, config):
    train_l_sum, valid_l_sum = 0, 0
    k = config.num_fold
    for i in range(k):
        data = getKFoldData(k, i, X_train, y_train)
        # net = LinearNet(config)
        train_ls, valid_ls = train(net, *data, config)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True