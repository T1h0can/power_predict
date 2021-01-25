from fitparse import FitFile
import pandas as pd
import os
import glob
from collections import defaultdict
import numpy as np

FIELDS_DATA = ['timestamp', 'speed', 'cadence', 'heart_rate', 'temperature', 'distance', 'altitude', 'power']


def _check_file_name(filename):
    if isinstance(filename, str):
        if filename.endswith('.fit'):
            if os.path.isfile(filename):
                return filename
            else:
                raise ValueError('The file does not exist.')
        else:
            raise ValueError('The file is not a fit file.')
    else:
        raise ValueError('filename needs to be a string. Got {}'.format(type(filename)))


def load_fit(filename):
    filename = _check_file_name(filename)
    records = FitFile(filename).get_messages('record')
    records_dict = defaultdict(list)
    for record in records:
        for key in FIELDS_DATA:
            value = record.get_value(key)
            records_dict[key].append(np.nan if value is None else value)
    records_df = pd.DataFrame.from_dict(records_dict)
    return records_df


def all_fit_to_csv(path):
    if not os.path.exists(path):
        raise ValueError('The dataset path is not exist')
    path = os.path.join(path, '*', '*.fit')
    files = glob.glob(path)
    # TODO: 可以并行
    for file in files:
        df = load_fit(file)
        dir_name = file[12: -20]
        out_path = os.path.join('dataset', 'csv', dir_name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        file_name = out_path + file[-20: -3] + 'csv'
        print(file_name)
        df.to_csv(file_name, index=False, sep=',')


if __name__ == '__main__':
    data_path = os.path.join('dataset', 'fit')#, '*', '*.fit')
    # print(data_path)
    if os.path.exists(data_path):
        print('True')
    else:
        print('False')
    all_fit_to_csv(data_path)
