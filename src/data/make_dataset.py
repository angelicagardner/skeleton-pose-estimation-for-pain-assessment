import argparse
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


def main(input_filepath, output_filepath):
    root_dir = Path(__file__).parent.parent.parent
    data_dir = root_dir / 'data'
    input_filepath = data_dir / input_filepath
    output_filepath = data_dir / output_filepath
    # 1. Read ground truth data
    ground_truth_file = input_filepath / 'movement_pain.csv'
    ground_truth = pd.read_csv(ground_truth_file, sep=';')
    for index, row in ground_truth.iterrows():
        # 2. Read Skeleton data
        for folder in input_filepath.iterdir():
            if str(folder.name) == row['scan_id']:
                df = pd.read_csv(folder / 'skeleton.csv')
                df['pain_level'] = row['pain_area'] + \
                    ' - Level ' + str(row['level'])
                # 3. Save as processed data
                i = 1
                for processed_file in (output_filepath / 'skeleton').iterdir():
                    if processed_file.name == str(folder.name) + '_1.csv':
                        if i < 2:
                            i = 2
                    elif processed_file.name == str(folder.name) + '_2.csv':
                        if i < 3:
                            i = 3
                    elif processed_file.name == str(folder.name) + '_3.csv':
                        if i < 4:
                            i = 4
                    elif processed_file.name == str(folder.name) + '_4.csv':
                        if i < 5:
                            i = 5
                    elif processed_file.name == str(folder.name) + '_5.csv':
                        if i < 6:
                            i = 6
                    elif processed_file.name == str(folder.name) + '_6.csv':
                        if i < 7:
                            i = 7
                    elif processed_file.name == str(folder.name) + '_7.csv':
                        if i < 8:
                            i = 8
                    elif processed_file.name == str(folder.name) + '_8.csv':
                        if i < 9:
                            i = 9
                file_name = str(folder.name) + '_' + str(i) + '.csv'
                df.to_csv(
                    output_filepath / 'skeleton' / file_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=str, default='raw')
    parser.add_argument('--output_filepath', type=str, default='processed')
    args = parser.parse_args()

    main(args.input_filepath, args.output_filepath)
