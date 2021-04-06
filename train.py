import torch
import torch.utils.data as Data
import torch.optim as optim
from load_data import load_data_for_linear_1
from models.Linear import LinearNet
from utils import seed_torch, k_fold, log_rmse
from sklearn.preprocessing import MinMaxScaler


def train_linear(net, config):
    m = MinMaxScaler()
    X_train, y_train, _, _, X_test, y_test = load_data_for_linear_1(conf, m)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    k_fold(X_train, y_train, net, conf)
    with torch.no_grad():
        y_pred = net(X_test)
        loss = torch.nn.MSELoss()
        # clipped_preds = torch.max(y_pred, torch.tensor(1.0))
        # print(clipped_preds)
        # rmse = torch.sqrt(loss(clipped_preds.log(), y_test_t.view(-1, 1).log()))
        # print(rmse.item())
        l = loss(y_pred, y_test_t)
        print(l.item())
        y_pred = y_pred.detach().numpy()
        # y_pred = m.inverse_transform(y_pred)
        print(y_pred)


if __name__ == '__main__':
    from config import config as conf
    seed_torch(conf.random_seed)
    # train(conf)
    # train_linear(conf)
    net = LinearNet(conf)
    train_linear(net, conf)
    # m = MinMaxScaler()
    # X_train, y_train, _, _, X_test, y_test = load_data_for_linear_1(conf, m)
    # X_train = torch.tensor(X_train.values, dtype=torch.float32)
    # y_train = torch.tensor(y_train.values, dtype=torch.float32)
    # X_test = torch.tensor(X_test.values, dtype=torch.float32)
    # y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
    # k_fold(X_train, y_train, net, conf)
    # with torch.no_grad():
    #     y_pred = net(X_test)
    #     loss = torch.nn.MSELoss()
    #     clipped_preds = torch.max(y_pred, torch.tensor(1.0))
    #     l = loss(y_pred, y_test_t)
    #     print(l.item())
    #     y_pred = y_pred.detach().numpy()
    #     y_pred = m.inverse_transform(y_pred)
    #     print(y_pred)
