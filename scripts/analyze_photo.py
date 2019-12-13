import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from molstove import tools
from molstove.parser import OrcaParser
from molstove.properties import calibrate_homo, calibrate_lumo, calculate_scharber_props

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


# # From: http://www.rsc.org/suppdata/ee/c3/c3ee42756k/c3ee42756k.pdf
# reference = {
#     'c1sc(-c2ccc(cn2)-c2[SiH2]c(cc2)-c2nccc3nsnc23)c2sccc12': 11.13,
#     'c1cnc(-c2cnc3c(c2)c2=C[SiH2]C=c2c2ccc4cscc4c32)c2nsnc12': 11.13,
#     'c1cc2ncc3c4c5cocc5c(cc4c4=C[SiH2]C=c4c3c2o1)-c1nccc2nsnc12': 11.12,
#     'c1ccc(-c2cc3c4nsnc4c4ccc5=C[SiH2]C=c5c4c3c3nsnc23)c2=C[SiH2]C=c12': 11.12,
# }


def recalculate_scharber(row: pd.Series, directory: str):
    calc_dir = os.path.join(directory, row['path'])
    p = OrcaParser(directory=calc_dir, output_file_name='orca.output')
    p.sanity_check()
    if 'UHF' in row['method']:
        orbitals = p.get_last_open_shell_orbitals()
        homo, lumo = tools.get_homo_lumo_energies(orbitals, is_open_shell=True)
    else:
        orbitals = p.get_last_orbitals()
        homo, lumo = tools.get_homo_lumo_energies(orbitals, is_open_shell=False)
    homo_calibrated = calibrate_homo(homo=homo * tools.EV_PER_HARTREE, method=row['method'])
    lumo_calibrated = calibrate_lumo(lumo=lumo * tools.EV_PER_HARTREE, method=row['method'])
    props = calculate_scharber_props(homo=homo_calibrated, lumo=lumo_calibrated, e_charge_sep=0)
    return props.pce


def main(directory: str) -> None:
    # Collect all JSON files
    json_files = glob.glob(os.path.join(directory, 'report_*.json'))

    frames = []
    for file in json_files:
        with open(file) as f:
            data = json.load(f)
            frames.append(pd.DataFrame(data))

    df = pd.concat(frames)

    # Determine PCE using different settings
    df['pce_alt'] = df.apply(func=lambda row: recalculate_scharber(row, directory=directory), axis='columns')

    # Join with reference data
    reference_series = pd.Series(reference)
    reference_series.name = 'pce_ref'

    df = df.join(reference_series, how='left', on='smiles')
    df = df.reset_index()

    # Compute mean
    df = df.groupby('smiles').agg(np.mean)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'max_colwidth',
                           200):
        print(df[['pce', 'pce_alt', 'pce_ref']])


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
