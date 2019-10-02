import argparse
from typing import List

from molstove import tools, properties


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smiles',
        help='SMILES string(s) (comma-separated)',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--seed',
        help='random seed',
        type=int,
        default=42,
        required=False,
    )
    parser.add_argument(
        '--max_num_conformers',
        help='maximum number of conformers to be generated',
        type=int,
        default=25,
        required=False,
    )
    parser.add_argument(
        '--rmsd_threshold',
        help='minimum RMSD for two conformers to be considered different (in Angstrom)',
        type=float,
        default=0.2,
        required=False,
    )
    parser.add_argument(
        '--delta_e_threshold',
        help='minimum energy difference between for two conformers to be considered different (in kcal/mol)',
        type=float,
        default=0.5,
        required=False,
    )
    parser.add_argument(
        '--energy_window',
        help='maximum energy difference the most stable conformer and any other conformer (in kcal/mol)',
        type=float,
        default=5.0,
        required=False,
    )
    parser.add_argument(
        '--max_num_opt_conformers',
        help='maximum number of conformers to be optimized',
        type=int,
        default=10,
        required=False,
    )
    parser.add_argument(
        '--basis_set',
        help='basis set employed in QC calculations',
        type=str,
        default='def2-SVP',
        required=False,
    )
    parser.add_argument(
        '--xc_functional',
        help='XC functional employed in QC calculations',
        type=str,
        default='PBE',
        required=False,
    )
    return parser


def hook() -> None:
    parser = get_parser()
    args = parser.parse_args()

    report_collection: List[dict] = []
    for smiles in args.smiles.split('.'):
        reports = properties.compute_pv_props(
            smiles=smiles,
            seed=args.seed,
            max_num_conformers=args.max_num_conformers,
            rmsd_threshold=args.rmsd_threshold,
            delta_e_threshold=args.delta_e_threshold,
            energy_window=args.energy_window,
            max_num_opt_conformers=args.max_num_opt_conformers,
            basis_set=args.basis_set,
            xc_functional=args.xc_functional,
        )

        report_collection += reports

    tools.write_to_json(report_collection, 'results.json')


if __name__ == '__main__':
    hook()
