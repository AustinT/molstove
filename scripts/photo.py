import dataclasses
import json
from typing import List, Optional

import numpy as np
from rdkit.Chem import Mol
from tqdm import tqdm

from molstove import tools, conformers, orca, properties
from molstove.properties import calibrate_homo, calibrate_lumo, ScharberResults
from molstove.tools import Atoms


@dataclasses.dataclass
class CalculationSettings:
    method: str
    basis_set: str
    aux_basis_set: Optional[str] = None
    open_shell: bool = False
    num_processors: int = 4


def generate_conformers(mol: Mol) -> List[Atoms]:
    # Generate conformers, optimize them, and collect clusters
    conformers.generate_conformers(mol, max_num_conformers=100, seed=42)
    energies = conformers.minimize_conformers(mol)
    conformer_list = conformers.collect_clusters(
        mol=mol,
        energies=energies,
        rmsd_threshold=0.1,
        delta_e_threshold=2,
        energy_window=10,
        max_num_conformers=5,
    )

    # Convert conformers to PySCF format
    return [tools.conformer_to_atoms(mol=mol, conformer=conformer) for conformer in conformer_list]


def optimize_structure(atoms: Atoms, charge: int, spin_multiplicity: int, settings: CalculationSettings) -> Atoms:
    c = orca.StructureOptCalculator(
        atoms=atoms,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        basis=settings.basis_set,
        aux_basis=settings.aux_basis_set,
        method=settings.method,
        open_shell=settings.open_shell,
        num_processes=settings.num_processors,
        directory=tools.create_tmp_dir_name(),
    )
    c.run()
    opt_results = c.parse_results()
    return opt_results.atoms


def compute_scharber_properties(atoms: Atoms, charge: int, spin_multiplicity: int, settings: CalculationSettings,
                                opt_method: str) -> dict:
    c = orca.SinglePointCalculator(
        atoms=atoms,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        basis=settings.basis_set,
        aux_basis=settings.aux_basis_set,
        method=settings.method,
        open_shell=settings.open_shell,
        num_processes=settings.num_processors,
        directory=tools.create_tmp_dir_name(),
    )
    c.run()
    results = c.parse_results()

    method = f'{opt_method}//{settings.method}/{settings.basis_set}'
    scharber_results = get_calibrated_scharber_predictions(method=method, homo=results.homo, lumo=results.lumo)
    report = {
        'path': str(c.directory),
        'method': method,
    }
    report.update(dataclasses.asdict(scharber_results))
    return report


def get_calibrated_scharber_predictions(method: str, homo: float, lumo: float) -> ScharberResults:
    try:
        homo_calibrated = calibrate_homo(homo=homo * tools.EV_PER_HARTREE, method=method)
        lumo_calibrated = calibrate_lumo(lumo=lumo * tools.EV_PER_HARTREE, method=method)
        return properties.calculate_scharber_props(homo=homo_calibrated, lumo=lumo_calibrated)
    except RuntimeError as e:
        print(f'Method {method} returned the following error when calculating Scharber properties: {e}')
        return ScharberResults(pce=np.nan, voc=np.nan, jsc=np.nan, ff=np.nan, eqe=np.nan, lumo_acceptor=np.nan,
                               e_charge_sep=np.nan, e_empirical_loss=np.nan)


def compute_hopv_props(smiles: str, num_processors: int) -> List[dict]:
    # Generate molecule from SMILES
    mol = tools.mol_from_smiles(smiles)
    charge = tools.get_molecular_charge(mol)
    spin_multiplicity = 1

    confs = generate_conformers(mol)

    reports = []

    # Run QC single point calculations on MM optimized structures
    settings_list = [
        CalculationSettings(method='BP86', basis_set='def2-SVP', aux_basis_set='def2/J', num_processors=num_processors),
        CalculationSettings(method='BP86', basis_set='STO-6G', num_processors=num_processors),
    ]

    # Optimize conformers
    for conf in confs:
        for settings in settings_list:
            results = compute_scharber_properties(atoms=conf,
                                                  charge=charge,
                                                  spin_multiplicity=spin_multiplicity,
                                                  settings=settings,
                                                  opt_method='MM')
            reports.append(results)

    opt_settings = CalculationSettings(method='BP86',
                                       basis_set='def2-SVP',
                                       aux_basis_set='def2/J',
                                       num_processors=num_processors)

    opt_confs = []
    for conf in confs:
        opt_confs.append(
            optimize_structure(atoms=conf, charge=charge, spin_multiplicity=spin_multiplicity, settings=opt_settings))

    # Run QC single point calculations on QC optimized structures
    settings_list = [
        CalculationSettings(method=method, basis_set='def2-SVP', aux_basis_set='def2/J', num_processors=num_processors)
        for method in ['BP86', 'B3LYP', 'PBE0', 'BHANDHLYP', 'M062X', 'RHF']
    ]

    settings_list.append(
        CalculationSettings(method='UHF',
                            basis_set='def2-SVP',
                            aux_basis_set='def2/J',
                            open_shell=True,
                            num_processors=num_processors))

    settings_list.append(
        CalculationSettings(method='BP86', basis_set='def2-TZVP', aux_basis_set='def2/J',
                            num_processors=num_processors))

    opt_method = f'{opt_settings.method}/{opt_settings.basis_set}'
    for conf in opt_confs:
        for settings in settings_list:
            results = compute_scharber_properties(atoms=conf,
                                                  charge=charge,
                                                  spin_multiplicity=spin_multiplicity,
                                                  settings=settings,
                                                  opt_method=opt_method)
            reports.append(results)

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

    # From: http://www.rsc.org/suppdata/ee/c3/c3ee42756k/c3ee42756k.pdf
    # smiles_list = [
    #     'c1sc(-c2ccc(cn2)-c2[SiH2]c(cc2)-c2nccc3nsnc23)c2sccc12',  # 11.13
    #     'c1cnc(-c2cnc3c(c2)c2=C[SiH2]C=c2c2ccc4cscc4c32)c2nsnc12',  # 11.13
    #     'c1cc2ncc3c4c5cocc5c(cc4c4=C[SiH2]C=c4c3c2o1)-c1nccc2nsnc12',  # 11.12
    #     'c1ccc(-c2cc3c4nsnc4c4ccc5=C[SiH2]C=c5c4c3c3nsnc23)c2=C[SiH2]C=c12',  # 11.12
    # ]

    for i, smiles in enumerate(tqdm(smiles_list)):
        print(smiles, flush=True)
        try:
            reports = compute_hopv_props(smiles, num_processors=num_processors)
            for item in reports:
                item['smiles'] = smiles
            with open(f'report_{i}.json', mode='w') as f:
                json.dump(reports, f)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
