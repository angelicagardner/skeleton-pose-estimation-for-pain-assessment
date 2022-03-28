import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelBinarizer


def get_feature_names(modality):
    dir = Path(__file__).parent.parent.parent / \
        'data' / 'processed' / 'train' / modality
    for file in dir.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return df.columns.values


def get_class_names(modality):
    pass


def load_file(file):
    df = pd.read_csv(file, header=0)
    labels = df['pain']
    df = df.drop(columns=['pain'])
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df.to_numpy(), labels


def load_dataset(input_filepath, modality):
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / 'data'
    train_filepath = data_dir / input_filepath / 'train' / modality
    test_filepath = data_dir / input_filepath / 'test' / modality
    # 1. Load train data
    X = list()
    y = list()
    for file in train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, labels = load_file(file)
            X.append(x)
            y.append(labels[0])
    n_length = X[0].shape[0]
    n_features = X[0].shape[1]
    X = np.array(X)
    X_train = X.reshape((len(X), 1, n_length, n_features))
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y)
    # 2. Load test data
    X = list()
    y = list()
    for file in test_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, labels = load_file(file)
            X.append(x)
            y.append(labels[0])
    n_length = X[0].shape[0]
    n_features = X[0].shape[1]
    X = np.array(X)
    X_test = X.reshape((len(X), 1, n_length, n_features))
    y_test = lb.transform(y)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, default='processed')
    parser.add_argument('--modality', type=str, default='skeleton')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_dataset(
        args.input_filepath, args.modality)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
