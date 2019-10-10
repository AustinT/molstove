import argparse
import glob
import os

import pandas as pd

from molstove import properties, tools

reference = {
    'c1coc(c1)-c1cc2sc3c(c4c[nH]cc4c4ccc5cscc5c34)c2c2ccccc12': 2.110,
    '[SiH2]1C=c2c3cc[nH]c3c3ccc4cc(-c5scc6cc[nH]c56)c5cscc5c4c3c2=C1': 2.620,
    'C1cc2ccc3c4cocc4c-4c(-[o]c5ccc6ccccc6c-45)c3c2c1': 3.330,
    'C1C=c2ccc3[nH]c4c(ncc5cc(-c6ccccn6)c6=CCC=c6c45)c3c2=C1': 3.090,
    'c1cc2c(scc2[nH]1)-c1cc2c3cocc3c3c4occc4sc3c2c2nsnc12': 4.620,
    '[SiH2]1C=c2c3ccsc3c3[se]c4cc(ncc4c3c2=C1)-c1ccc[se]1': 6.500,
    'C1C=c2ccc3c4cscc4c4c5ncc(cc5[nH]c4c3c2=C1)-c1scc2sccc12': 4.170,
    'C1C=c2ccc3ncc4cc5cc(sc5cc4c3c2=C1)-c1scc2C=CCc12': 4.140,
    'C1c2cc([nH]c2-c2sc3ccncc3c12)-c1scc2[nH]ccc12': 0.530,
    'C1C=Cc2c1csc2-c1cc2cnc3c4[nH]ccc4c4=C[SiH2]C=c4c3c2c2=C[SiH2]C=c12': 6.100,
}


def main(directory: str) -> None:
    # Collect all JSON files
    json_files = glob.glob(os.path.join(directory, 'report_*.json'))

    frames = []
    for file in json_files:
        frames.append(pd.read_json(file))

    df = pd.concat(frames)

    df['pce*'] = df.apply(
        axis='columns',
        func=lambda r: properties.calculate_scharber_props(
            homo=r['homo'] * tools.EV_PER_HARTREE,
            lumo=r['lumo'] * tools.EV_PER_HARTREE,
            eqe=1.0,
            ff=1.0,
            e_empirical_loss=0.0,
        ).pce,
    )

    # Join with reference data
    reference_series = pd.Series(reference)
    reference_series.name = 'pce_ref'

    df = df.join(reference_series, how='left', on='smiles')
    df = df.reset_index()

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'max_colwidth',
                           100):
        print(df[['smiles', 'xc', 'pce', 'pce*', 'pce_ref', 'elapsed']])


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
