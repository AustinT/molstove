import argparse
import dataclasses
from typing import Dict, Any, List

from molstove import tools, conformers, calculator, properties


def generate_report(c: calculator.Calculator) -> Dict[str, Any]:
    homo, lumo = c.get_homo_lumo()
    return {
        'atoms': c.mol.atom,
        'charge': c.mol.charge,
        'spin_multiplicity': c.mol.spin,
        'basis': c.mol.basis,
        'xc_functional': c.mf.xc,
        'energy': c.get_energy(),
        'homo': homo,
        'lumo': lumo,
    }


def main(args) -> List[dict]:
    # Generate molecule from SMILES
    mol = tools.mol_from_smiles(args.smiles)

    # Generate conformers
    conformers.generate_conformers(mol, max_num_conformers=args.max_num_conformers, seed=args.seed)
    energies = conformers.minimize_conformers(mol)
    conformer_list = conformers.collect_clusters(
        mol=mol,
        energies=energies,
        rmsd_threshold=args.rmsd_threshold,
        delta_e_threshold=args.delta_e_threshold,
        energy_window=args.energy_window,
        max_num_conformers=args.max_num_opt_conformers,
    )

    # Convert conformers to PySCF format
    atoms_list = [tools.conformer_to_atoms(mol=mol, conformer=conformer) for conformer in conformer_list]
    charge = tools.get_molecular_charge(mol)

    reports = []

    # Run QC calculations and get predictions based on Scharber model
    for i, atoms in enumerate(atoms_list):
        try:
            c = calculator.Calculator(atoms=atoms, charge=charge, basis=args.basis_set, xc=args.xc_functional)
            atoms_opt = calculator.optimize(c)

            c2 = calculator.Calculator(atoms=atoms_opt, charge=charge, basis=args.basis_set, xc=args.xc_functional)
            c2.run()

            homo, lumo = c2.get_homo_lumo()
            scharber = properties.calculate_scharber_props(homo=homo, lumo=lumo)

            report = generate_report(c2)
            report.update(dataclasses.asdict(scharber))
            report['smiles'] = args.smiles

            tools.write_to_json(report, path=f'report_{i}.json')

            reports.append(report)

        except RuntimeError as e:
            print(f'Calculations for conformer {i} failed: {e}')

    tools.write_to_json(reports, path='report.json')

    return reports


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
    main(args)


if __name__ == '__main__':
    hook()
