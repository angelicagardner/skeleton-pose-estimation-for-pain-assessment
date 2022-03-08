import argparse
import pandas as pd
import numpy as np
from numpy import dstack
from pathlib import Path


def load_file(file):
    df = pd.read_csv(file, header=0)
    y = df['pain_level']
    df.drop(df.columns[-1], axis=1, inplace=True)
    return df.to_numpy(), y


def main(input_filepath):
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / 'data'
    input_filepath = data_dir / input_filepath / 'train' / 'skeleton'
    loaded_X = list()
    loaded_Y = list()
    for file in input_filepath.iterdir():
        if file.is_file() and file.name.endswith('.csv'):
            x, y = load_file(file)
            loaded_X.append(x)
            loaded_Y.append(y)
    X = dstack(loaded_X)
    Y = np.array(loaded_Y)
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, default='processed')
    args = parser.parse_args()

    main(args.input_filepath)
