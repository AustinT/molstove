import argparse
import glob
import os

import pandas as pd


def main(directory: str) -> None:
    # Collect all JSON files
    json_files = glob.glob(os.path.join(directory, 'report_*.json'))

    frames = []
    for file in json_files:
        frames.append(pd.read_json(file))

    df = pd.concat(frames)

    # Remove unwanted columns
    df = df.drop(columns=[
        'atoms', 'charge', 'spin_multiplicity', 'basis_set', 'lumo_acceptor', 'e_charge_sep', 'e_empirical_loss', 'ff',
        'eqe', 'path', 'elapsed'
    ])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        help='directory containing JSON files',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.directory)
