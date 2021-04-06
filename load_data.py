import os
import glob
import pandas as pd
import torch
import numpy as np
import fit_to_csv
import random
from sklearn.preprocessing import StandardScaler


def load_data(config):
    """
    加载所有数据到df，扔掉小于3000行的，返回df的list
    :return: list of all df
    """
    csv_path = os.path.join('dataset', 'csv')
    if not os.path.exists(csv_path):
        fit_path = os.path.join('dataset', 'fit')
        fit_to_csv.all_fit_to_csv(fit_path)
    csv_path = os.path.join(csv_path, '*', '*.csv')
    # csv_path = os.path.join(csv_path, '*.csv')
    # print(csv_path)
    csv_files = glob.glob(csv_path)
    # all_features, all_labels = [], []
    all_df = []
    n = 0
    for file in csv_files:
        # print(file[12: -21])
        df = pd.read_csv(file)
        # print(df.shape[0])
        if df.shape[0] < 3000:
            continue
        n += 1
        # 调试
        print(file)
        # df.astype(float32)
        # calc_power_sealevel(df)
        calc_power_sealevel(df, config.althletes_weight[file[12:-21]])
        calc_grade(df)
        calc_hr_grade(df)
        calc_cad_grade(df)
        calc_acceleration(df)
        fill_and_drop(df)
        df = df.reset_index(drop=True)
        # std_features(df)
#        features_cols = df.columns.values.tolist()[config.features_start: config.features_end]
        # print(features_cols)
#        df[features_cols] = m.fit_transform(df[features_cols])
        # features_cols = ['speed', 'cadence', 'heart_rate', 'temperature']
        # df[features_cols] = df[features_cols].apply(lambda x: (x - x.mean()) / (x.std()))
        '''
        df[features_cols] = df[features_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        '''
        # print(df.iloc[:5, config.features_start: config.features_end])
        # print(df.info())
        # if file == 'dataset/csv/ZhouZheng/2019-08-28-10-06.csv':
        #     print(df.iloc[:5, 4: 9])
        #     print(df.iloc[-5:, 4: 9])
        # ss = StandardScaler()
        # scale_features = df.columns.values.tolist()[1:9]
        # df[scale_features] = ss.fit_transform(df[scale_features])
        # print(df.iloc[:5, 1: 9])
        all_df.append(df)
        # features = df.iloc[:, 1: 9]
        # std_features(features)
        # print(df.iloc[:5, 1: 9])
        # print(features.head())
        # labels = df.iloc[:, -1]
        # all_features.append(features)
        # all_labels.append(labels)
    print('{} files appended'.format(n))
    return all_df


def std_features(df: pd.DataFrame):
    col = df.columns.values.tolist()[1:9]
    # for key in ['speed', 'cadence', 'heart_rate', 'temperature']:
    for key in col:
        values = df[key]
        mean = values.mean()
        std = values.std()
        values = (values - mean) / std
        df[key] = values


def calc_power_sealevel(df: pd.DataFrame, weight):
    """
    计算海平面功率
    Based on research from Bassett, D.R. Jr., C.R. Kyle, L. Passfield, J.P. Broker, and E.R. Burke.
    31:1665-76, 1999.
    Note we assume the athlete is acclimatized for simplicity.
    acclimated:
    vo2maxPct = -1.1219 * km ** 2 - 1.8991 * km + 99.921
    R^2 = 0.9729
    unacclimated:
    v02maxPct = 0.1781 * km ** 3 - 1.434 * km ** 2 - 4.0726 ** km + 100.35
    R^2 = 0.9739
    altitude(km)
    按在对应海拔高度经过适应的
    :param df: pandas.DataFrame
    """
    alt_km = df['altitude'] / 1000
    vo2max_adjust = (-1.1219 * alt_km ** 2 - 1.8991 * alt_km + 99.921) / 100
    # power_sl = round(df['power'] / vo2max_adjust / weight, 2)
    power_sl = round(df['power'] / vo2max_adjust, 2)
    df['sea_level_power'] = power_sl


def calc_grade(df: pd.DataFrame, periods=2):
    """
    计算坡度
    :param df: pd.DataFrame
    :param periods: int 计算时的偏移
    """
    diff_altitude = df['altitude'].diff(periods=periods)
    diff_distance = df['distance'].diff(periods=periods)
    grade = round(diff_altitude / diff_distance * 100, 2)
    # print(type(grade))
    grade[:periods] = 0.0
    # df['grade'] = grade
    df.insert(5, 'grade', grade)
    df[df['grade'] == np.inf] = np.nan
    df[df['grade'] == -np.inf] = np.nan


def calc_acceleration(df: pd.DataFrame, periods=1):
    """
    计算加速度
    :param df: pd.DataFrame
    :param periods: int 计算时的偏移
    :return:
    """
    acceleration = round(df['speed'].diff(periods=periods) / periods, 3)
    acceleration[:periods] = 0.0
    df.insert(2, 'acceleration', acceleration)


def calc_hr_grade(df: pd.DataFrame, periods=1):
    """
    计算心率变化率
    :param df: pd.DataFrame
    :param periods: int 计算时的偏移
    :return:
    """
    hr_grade = df['heart_rate'].diff(periods=periods)
    hr_grade[:periods] = 0
    df.insert(4, 'hr_grade', hr_grade)


def calc_cad_grade(df: pd.DataFrame, periods=1):
    """
    计算踏频变化率
    :param df: pd.DataFrame
    :param periods: int 计算时的偏移
    :return:
    """
    cad_grade = df['cadence'].diff(periods=periods)
    cad_grade[:periods] = 0
    df.insert(3, 'cad_grade', cad_grade)


def fill_and_drop(df: pd.DataFrame):
    """
    大于2200或等于0的功率、大于220或小于50的心率、等于0的速度踏频距离全用nan填充
    然后drop掉
    drop all 0 power,speed, cadence, distance
    drop all power data which >2200
    drop all hr data which <50 or >220
    :param df:
    :return:
    """
    df[df['power'] > 2200] = np.nan
    df[df['power'] == 0] = np.nan
    df[df['heart_rate'] > 220] = np.nan
    df[df['heart_rate'] < 50] = np.nan
    df[df['cadence'] == 0] = np.nan
    df[df['distance'] == 0.0] = np.nan
    df[df['speed'] == 0.0] = np.nan
    # df[df['grade'] == np.inf] = np.nan
    # df[df['grade'] == -np.inf] = np.nan
    df.dropna(axis=0, how='any', inplace=True)


def splite_dataset(dataset, isLinear=False):
    # TODO: 咋地划分数据集，直接分一个文件做测试集，验证集训练集28开
    test_index = np.random.randint(0, len(dataset))
    test_set = dataset[test_index]
    dataset.pop(test_index)
    if isLinear:
        dataset = pd.concat(dataset)
    shuffled_indices = np.random.permutation(len(dataset))
    valid_set_size = int(len(dataset) * 0.2)
    valid_indices = shuffled_indices[:valid_set_size]  # 20%验证集
    train_indices = shuffled_indices[valid_set_size:]
    train_set = dataset.iloc[train_indices]
    valid_set = dataset.iloc[valid_indices]
    return train_set, valid_set, test_set


def load_data_for_linear_1(config, m):
    all_df = load_data(config)
    for df in all_df:
        features_cols = df.columns.values.tolist()[config.features_start: config.features_end]
        # print(features_cols)
        df[features_cols] = m.fit_transform(df[features_cols])
    # print(type(all_df))
    train_set, valid_set, test_set = splite_dataset(all_df, isLinear=True)
    train_features, train_labels = train_set.iloc[:, config.features_start: config.features_end], train_set.iloc[:, [config.labels_index]]
    valid_features, valid_labels = valid_set.iloc[:, config.features_start: config.features_end], valid_set.iloc[:, [config.labels_index]]
    test_features, test_labels = test_set.iloc[:, config.features_start: config.features_end], test_set.iloc[:, [config.labels_index]]
    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels


def load_data_for_linear(config):
    """
    返回tensor
    :param config:
    :return:
    """
    all_df = load_data(config)
    # k = len(all_df)
    all_df = pd.concat(all_df)
    print(all_df.shape)
    all_df = all_df.sample(frac=1.0)
    test_size = int(0.1 * all_df.shape[0])
    test_df = all_df.iloc[:test_size]
    train_df = all_df.iloc[test_size:]
    train_features = train_df.iloc[:, config.features_start: config.features_end]
    train_labels = train_df.iloc[:, [config.labels_index]]
    test_features = test_df.iloc[:, config.features_start: config.features_end]
    test_labels = test_df.iloc[:, [config.labels_index]]
    print(train_features.shape, train_labels.shape, test_features.shape, test_labels.shape)
    return train_features, train_labels, test_features, test_labels
    # all_features = all_df.iloc[:, config.features_start: config.features_end]
    # all_labels = all_df.iloc[:, config.labels_index]
    # print(all_features.shape, all_labels.shape)
    # from sklearn.model_selection import train_test_split
    # X_train, y_train, X_test, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)
    # return X_train, y_train, X_test, y_test
    # all_features = torch.tensor(all_features.values, dtype=torch.float32)
    # all_labels = torch.tensor(all_labels.values, dtype=torch.float32)
    # return all_features, all_labels


def load_data_for_rnn(config):
    # TODO: 把数据集处理成时间序列
    '''
    all_df [m * n * 13] m个文件，每个n行，13列
    features [m * n * 8] m个文件，每个n行，8列
    labels [m * n * 1] m个文件，每个n行，1列
    ->
    features [m * (n - t) * t * 8] m个文件，每个n-t组时间序列数据，每组时间序列t行， 8列
    labels [m * (n - t) * t * 1] m个文件，每个n-t组时间序列数据，每组时间序列t行， 1列
    for i in range(m) yield features[i], labels[i]
    :return:
    '''
    all_df = load_data(config)
    ts = []
    for df in all_df:
        nums = df.shape[0] - config.time_step
        print(nums)
        for i in range(nums):
            ts.append(df.iloc[i: i + config.time_step])
    print(len(ts))
    for i in range(5):
        print(ts[i])


if __name__ == '__main__':
    from config import config as conf
    from sklearn.preprocessing import MinMaxScaler
    m = MinMaxScaler()
    # myconf = config.config
    load_data_for_linear(conf)

    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # X_train, y_train, X_test, y_test = load_data_for_linear(conf)
    # model.fit(X_train, y_train)
    #
    # train_score = model.score(X_train, y_train)
    # test_score = model.score(X_test, y_test)
    #
    # print(train_score, test_score)

    # load_data_for_rnn(conf)
