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


def get_class_names(modality, nopain=True, binary=False, level=False):
    train_filepath = data_dir / 'processed' / 'train' / modality
    y = list()
    for file in train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, labels = load_file(file, level)
            if not nopain:
                if labels[0] == 'No Pain':
                    continue
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


def load_file(file, level):
    df = pd.read_csv(file, header=0)
    if level:
        labels = df['pain_level']
    else:
        labels = df['pain_area']
    df = df.drop(columns=['pain_area', 'pain_level'])
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df, labels


def load_dataset(modality, nopain=True, binary=False, fusion=False, level=False, only_minority=False):
    train_filepath = data_dir / 'processed' / 'train' / modality
    test_filepath = data_dir / 'processed' / 'test' / modality
    if only_minority:
        nr_mild = 0
    # 1. Load train data
    X = list()
    y = list()
    for file in train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            if modality == 'skeleton' and fusion:
                if not (data_dir / 'processed' / 'train' /
                        'AUs' / file.name).is_file():
                    continue
            x, labels = load_file(file, level)
            if not nopain:
                if labels[0] == 'No Pain':
                    continue
            if only_minority:
                if (labels[0] == 'Mild' or labels[0] == 'Lower Body') and nr_mild > 2:
                    continue
                elif (labels[0] == 'Mild' or labels[0] == 'Lower Body'):
                    nr_mild += 1
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
            if modality == 'skeleton' and fusion:
                if not (data_dir / 'processed' / 'test' /
                        'AUs' / file.name).is_file():
                    continue
            x, labels = load_file(file, level)
            if not nopain:
                if labels[0] == 'No Pain':
                    continue
            if only_minority:
                if (labels[0] == 'Mild' or labels[0] == 'Lower Body') and nr_mild > 2:
                    continue
                elif (labels[0] == 'Mild' or labels[0] == 'Lower Body'):
                    nr_mild += 1
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


def load_fusioned_dataset(nopain=True, binary=False, level=False):
    body_train_filepath = data_dir / 'processed' / 'train' / 'skeleton'
    body_test_filepath = data_dir / 'processed' / 'test' / 'skeleton'
    face_train_filepath = data_dir / 'processed' / 'train' / 'AUs'
    face_test_filepath = data_dir / 'processed' / 'test' / 'AUs'
    # 1. Load train data
    X = list()
    y = list()
    for file in body_train_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            body_x, body_labels = load_file(file, level)
            if not nopain:
                if body_labels[0] == 'No Pain':
                    continue
            has_equivalent_face_file = False
            for second_file in face_train_filepath.iterdir():
                if second_file.is_file() and second_file.name.endswith('.csv') and second_file.name == file.name:
                    # Concatenate features
                    face_x, face_labels = load_file(second_file, level)
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
            body_x, body_labels = load_file(file, level)
            if not nopain:
                if body_labels[0] == 'No Pain':
                    continue
            body_x = body_x.to_numpy()
            has_equivalent_face_file = False
            for second_file in face_test_filepath.iterdir():
                if second_file.is_file() and second_file.name.endswith('.csv') and second_file.name == file.name:
                    # Concatenate features
                    face_x, face_labels = load_file(second_file, level)
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
