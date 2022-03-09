import argparse
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
    input_filepath = data_dir / input_filepath / 'train' / 'skeleton'
    dataX = list()
    dataY = list()
    for file in input_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, y = load_file(file)
            dataX.append(x)
            dataY.append(y[0])
    X = np.reshape(dataX, (len(dataX), dataX[0].shape[0], dataX[0].shape[1]))
    Y = pd.get_dummies(dataY)
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, default='processed')
    args = parser.parse_args()

    load_dataset(args.input_filepath)
