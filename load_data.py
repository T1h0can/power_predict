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
    csv_path = os.path.join(csv_path, 'Zhou*', '*.csv')
    # csv_path = os.path.join(csv_path, '*.csv')
    # print(csv_path)
    csv_files = glob.glob(csv_path)
    # all_features, all_labels = [], []
    all_df = []
    n = 0
    for file in csv_files:
        # print(file)
        df = pd.read_csv(file)
        # print(df.shape[0])
        if df.shape[0] < 3000:
            continue
        n += 1
        print(file)
        # df.astype(float32)
        calc_power_sealevel(df)
        calc_grade(df)
        calc_hr_grade(df)
        calc_cad_grade(df)
        calc_acceleration(df)
        fill_and_drop(df)
        df = df.reset_index(drop=True)
        # std_features(df)
        features_cols = df.columns.values.tolist()[config.features_start: config.features_end]
        df[features_cols] = df[features_cols].apply(lambda x: (x - x.mean()) / (x.std()))
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
    # if is_Linear:
    #     all_df = pd.concat(all_df)
    #     all_features = all_df.iloc[:, 1: 9]
    #     # all_features = pd.concat(all_features)
    #     all_labels = all_df.iloc[:, -1]
    #     # all_labels = pd.concat(all_labels)
    #     print(all_features.shape, all_labels.shape)
    #     return all_features, all_labels
    # else:
    #     for df in all_df:
    #         features = df.iloc[:, 1: 9]
    #         labels = df.iloc[:, -1]

        # features = torch.tensor(features.values, dtype=torch.float32)
        # labels = torch.tensor(labels.values, dtype=torch.float32)
        # yield features, labels


def std_features(df: pd.DataFrame):
    col = df.columns.values.tolist()[1:9]
    # for key in ['speed', 'cadence', 'heart_rate', 'temperature']:
    for key in col:
        values = df[key]
        mean = values.mean()
        std = values.std()
        values = (values - mean) / std
        df[key] = values


def calc_power_sealevel(df: pd.DataFrame):
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
    power_sl = round(df['power'] / vo2max_adjust, 2)
    # print(power_sl)
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
    # print(grade)
    grade[:periods] = 0.0
    # df['grade'] = grade
    df.insert(5, 'grade', grade)


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
    df[df['grade'] == np.inf] = np.nan
    df[df['grade'] == -np.inf] = np.nan
    df.dropna(axis=0, how='any', inplace=True)


def load_data_for_linear(config):
    """
    返回tensor
    :param config:
    :return:
    """
    # TODO: 随机抽出测试集，剩余为训练集
    all_df = load_data(config)
    k = len(all_df)

    all_df = pd.concat(all_df)
    all_features = all_df.iloc[:, config.features_start: config.features_end]
    all_labels = all_df.iloc[:, config.labels_index]
    print(all_features.shape, all_labels.shape)
    all_features = torch.tensor(all_features.values, dtype=torch.float32)
    all_labels = torch.tensor(all_labels.values, dtype=torch.float32)
    return all_features, all_labels


def load_data_for_rnn():
    # TODO: 把数据集处理成时间序列
    all_df = load_data()


if __name__ == '__main__':
    from config import config
    # myconf = config.config
    load_data_for_linear(config)
