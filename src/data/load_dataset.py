import argparse
from cgi import test
import pandas as pd
import numpy as np
from pathlib import Path


def load_file(file):
    df = pd.read_csv(file, header=0)
    labels = df['pain_level']
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df.to_numpy(), labels


def load_dataset(input_filepath):
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / 'data'
    train_filepath = data_dir / input_filepath / 'train' / 'skeleton'
    test_filepath = data_dir / input_filepath / 'test' / 'skeleton'
    # 1. Load train data
    dataX_train = list()
    datay_train = list()
    for file in train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, y = load_file(file)
            dataX_train.append(x)
            datay_train.append(y[0])
    trainX = np.reshape(
        dataX_train, (len(dataX_train), dataX_train[0].shape[0], dataX_train[0].shape[1]))
    trainy = pd.get_dummies(datay_train)
    # 2. Load test data
    dataX_test = list()
    datay_test = list()
    for file in test_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, y = load_file(file)
            dataX_test.append(x)
            datay_test.append(y[0])
    testX = np.reshape(
        dataX_test, (len(dataX_test), dataX_test[0].shape[0], dataX_test[0].shape[1]))
    testy = pd.get_dummies(datay_test)
    testy, tmp = testy.align(trainy, join='outer', axis=1, fill_value=0)
    return trainX, testX, trainy, testy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, default='processed')
    args = parser.parse_args()

    load_dataset(args.input_filepath)
