import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import LabelBinarizer


data_dir = Path(__file__).parent.parent.parent / 'data'


def get_feature_names(modality):
    dir = Path(__file__).parent.parent.parent / \
        'data' / 'processed' / 'train' / modality
    for file in dir.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            df = pd.read_csv(file)
            return df.columns.values


def get_class_names(modality, binary=False):
    train_filepath = data_dir / 'processed' / 'train' / modality
    y = list()
    for file in train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, labels = load_file(file)
            if binary:
                if labels[0] != 'No Pain':
                    y.append('Pain')
                else:
                    y.append('No Pain')
            else:
                y.append(labels[0])
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y)
    return lb.inverse_transform(y_train)


def load_file(file):
    df = pd.read_csv(file, header=0)
    labels = df['pain']
    df = df.drop(columns=['pain'])
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df, labels


def load_dataset(modality, binary=False):
    train_filepath = data_dir / 'processed' / 'train' / modality
    test_filepath = data_dir / 'processed' / 'test' / modality
    # 1. Load train data
    X = list()
    y = list()
    for file in train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, labels = load_file(file)
            x = x.to_numpy()
            X.append(x)
            if binary:
                if labels[0] != 'No Pain':
                    y.append('Pain')
                else:
                    y.append('No Pain')
            else:
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
            x = x.to_numpy()
            X.append(x)
            if binary:
                if labels[0] != 'No Pain':
                    y.append('Pain')
                else:
                    y.append('No Pain')
            else:
                y.append(labels[0])
    n_length = X[0].shape[0]
    n_features = X[0].shape[1]
    X = np.array(X)
    X_test = X.reshape((len(X), 1, n_length, n_features))
    y_test = lb.transform(y)
    return X_train, X_test, y_train, y_test


def load_fusioned_dataset(binary=False):
    body_train_filepath = data_dir / 'processed' / 'train' / 'skeleton'
    body_test_filepath = data_dir / 'processed' / 'test' / 'skeleton'
    face_train_filepath = data_dir / 'processed' / 'train' / 'AUs'
    face_test_filepath = data_dir / 'processed' / 'test' / 'AUs'
    # 1. Load train data
    X = list()
    y = list()
    for file in body_train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            body_x, body_labels = load_file(file)
            has_equivalent_face_file = False
            for second_file in face_train_filepath.iterdir():
                if second_file.is_file() and second_file.name.endswith('.csv') and second_file.name == file.name:
                    # Concatenate features
                    face_x, face_labels = load_file(second_file)
                    full_X = np.concatenate((body_x, face_x), axis=1)
                    body_x = body_x.to_numpy()
                    face_x = face_x.to_numpy()
                    X.append(full_X)
                    if binary:
                        if body_labels[0] != 'No Pain':
                            y.append('Pain')
                        else:
                            y.append('No Pain')
                    else:
                        y.append(body_labels[0])
                    has_equivalent_face_file = True
                    break
            if not has_equivalent_face_file:
                continue
    n_length = X[0].shape[0]
    n_features = X[0].shape[1]
    X = np.array(X)
    X_train = X.reshape((len(X), 1, n_length, n_features))
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y)
    # 2. Load test data
    X = list()
    y = list()
    for file in body_test_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            body_x, body_labels = load_file(file)
            body_x = body_x.to_numpy()
            has_equivalent_face_file = False
            for second_file in face_test_filepath.iterdir():
                if second_file.is_file() and second_file.name.endswith('.csv') and second_file.name == file.name:
                    # Concatenate features
                    face_x, face_labels = load_file(second_file)
                    face_x = face_x.to_numpy()
                    full_X = np.concatenate((body_x, face_x), axis=1)
                    X.append(full_X)
                    if binary:
                        if body_labels[0] != 'No Pain':
                            y.append('Pain')
                        else:
                            y.append('No Pain')
                    else:
                        y.append(body_labels[0])
                    has_equivalent_face_file = True
                    break
            if not has_equivalent_face_file:
                continue
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
