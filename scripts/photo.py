import dataclasses
import json
import time
from typing import List

from molstove import tools, conformers, orca, properties
from molstove.properties import calibrate_homo, calibrate_lumo


@dataclasses.dataclass
class CalculationSettings:
    method: str
    basis_set: str
    aux_basis_set: str
    open_shell: bool = False
    num_processors: int = 4


def compute_pv_props(
        smiles: str,
        opt_settings: CalculationSettings,
        sp_settings_list: List[CalculationSettings],
) -> List[dict]:
    # Generate molecule from SMILES
    mol = tools.mol_from_smiles(smiles)

    # Generate conformers
    conformers.generate_conformers(mol, max_num_conformers=25, seed=42)
    energies = conformers.minimize_conformers(mol)
    conformer_list = conformers.collect_clusters(
        mol=mol,
        energies=energies,
        rmsd_threshold=0.1,
        delta_e_threshold=5,
        energy_window=10,
        max_num_conformers=5,
    )

    # Convert conformers to PySCF format
    atoms_list = [tools.conformer_to_atoms(mol=mol, conformer=conformer) for conformer in conformer_list]
    charge = tools.get_molecular_charge(mol)
    spin_multiplicity = 1

    reports = []

    # Run QC calculations and get predictions based on Scharber model
    for i, atoms in enumerate(atoms_list):
        try:
            start_time = time.time()

            c = orca.StructureOptCalculator(
                atoms=atoms,
                charge=charge,
                spin_multiplicity=spin_multiplicity,
                basis=f'{opt_settings.basis_set} {opt_settings.aux_basis_set}',
                method=opt_settings.method,
                open_shell=opt_settings.open_shell,
                num_processes=opt_settings.num_processors,
            )
            c.run()
            opt_results = c.parse_results()
            opt_atoms = opt_results.atoms

            for sp_settings in sp_settings_list:
                c = orca.SinglePointCalculator(
                    atoms=opt_atoms,
                    charge=charge,
                    spin_multiplicity=spin_multiplicity,
                    basis=f'{sp_settings.basis_set} {sp_settings.aux_basis_set}',
                    method=sp_settings.method,
                    open_shell=sp_settings.open_shell,
                    num_processes=sp_settings.num_processors,
                )
                c.run()
                sp_results = c.parse_results()
                homo, lumo = sp_results.homo, sp_results.lumo

                method = f'{opt_settings.method}/{opt_settings.basis_set}//' \
                         f'{sp_settings.method}/{sp_settings.basis_set}'
                homo_calibrated = calibrate_homo(homo=homo * tools.EV_PER_HARTREE, method=method)
                lumo_calibrated = calibrate_lumo(lumo=lumo * tools.EV_PER_HARTREE, method=method)
                scharber = properties.calculate_scharber_props(homo=homo_calibrated, lumo=lumo_calibrated)

                report = {
                    'smiles': smiles,
                    'atoms': [dataclasses.asdict(atom) for atom in opt_atoms],
                    'charge': charge,
                    'spin_multiplicity': spin_multiplicity,
                    'basis_set': sp_settings.basis_set,
                    'basis_set_opt': opt_settings.basis_set,
                    'method': sp_settings.method,
                    'method_opt': opt_settings.method,
                    'homo': homo,
                    'lumo': lumo,
                    'energy': sp_results.energy,
                    'path': c.calculation_directory,
                }

                report.update(dataclasses.asdict(scharber))
                report['smiles'] = smiles
                report['elapsed'] = time.time() - start_time

                reports.append(report)

        except RuntimeError as e:
            print(f'Calculations for conformer {i} failed: {e}')

    return reports


def main():
    num_processors = 1

    smiles_list = [
        'c1coc(c1)-c1cc2sc3c(c4c[nH]cc4c4ccc5cscc5c34)c2c2ccccc12',
        '[SiH2]1C=c2c3cc[nH]c3c3ccc4cc(-c5scc6cc[nH]c56)c5cscc5c4c3c2=C1',
        'C1cc2ccc3c4cocc4c-4c(-[o]c5ccc6ccccc6c-45)c3c2c1',
        'C1C=c2ccc3[nH]c4c(ncc5cc(-c6ccccn6)c6=CCC=c6c45)c3c2=C1',
        'c1cc2c(scc2[nH]1)-c1cc2c3cocc3c3c4occc4sc3c2c2nsnc12',
        '[SiH2]1C=c2c3ccsc3c3[se]c4cc(ncc4c3c2=C1)-c1ccc[se]1',
        'C1C=c2ccc3c4cscc4c4c5ncc(cc5[nH]c4c3c2=C1)-c1scc2sccc12',
        'C1C=c2ccc3ncc4cc5cc(sc5cc4c3c2=C1)-c1scc2C=CCc12',
        'C1c2cc([nH]c2-c2sc3ccncc3c12)-c1scc2[nH]ccc12',
        'C1C=Cc2c1csc2-c1cc2cnc3c4[nH]ccc4c4=C[SiH2]C=c4c3c2c2=C[SiH2]C=c12',
    ]

    opt_settings = CalculationSettings(method='BP86',
                                       basis_set='def2-SVP',
                                       aux_basis_set='def2/J',
                                       num_processors=num_processors)

    sp_settings_list = [
        CalculationSettings(method=method, basis_set='def2-SVP', aux_basis_set='def2/J', num_processors=num_processors)
        for method in ['BP86', 'B3LYP', 'PBE0', 'BHANDHLYP', 'M062X', 'RHF']
    ]

    sp_settings_list.append(
        CalculationSettings(method='UHF',
                            basis_set='def2-SVP',
                            aux_basis_set='def2/J',
                            open_shell=True,
                            num_processors=num_processors))

    sp_settings_list.append(
        CalculationSettings(method='BP86', basis_set='def2-TZVP', aux_basis_set='def2/J',
                            num_processors=num_processors))

    for i, smiles in enumerate(smiles_list):
        try:
            report = compute_pv_props(smiles, opt_settings=opt_settings, sp_settings_list=sp_settings_list)
            with open(f'report_{i}.json', mode='w') as f:
                json.dump(report, f)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
