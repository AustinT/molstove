"""
Functions for doing DFT calculations in the same way as
the clean energy project
"""
import time
import dataclasses
from typing import List, Optional
from rdkit.Chem import Mol

from molstove import tools, conformers, orca
from molstove.properties import calibrate_homo, calibrate_lumo
from molstove.tools import Atoms


@dataclasses.dataclass
class CalculationSettings:
    """
    Stores the settings for DFT calculations
    """
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
    return [tools.conformer_to_atoms(mol=mol, conformer=conformer)
            for conformer in conformer_list]


def optimize_structure(atoms: Atoms, charge: int, spin_multiplicity: int,
                       settings: CalculationSettings) -> Atoms:
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
    print(f"Doing structure opt calc in {c.directory}", flush=True)
    t_start = time.time()
    c.run()
    t_end = time.time()
    print(f"Structure optimization took {(t_end - t_start)/60:.2f} min",
          flush=True)
    opt_results = c.parse_results()
    return opt_results.atoms


def compute_calibrated_homo_lumo(atoms: Atoms, charge: int,
                                spin_multiplicity: int,
                                settings: CalculationSettings,
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
    print(f"Doing single point calc in {c.directory}", flush=True)
    t_start = time.time()
    c.run()
    t_end = time.time()
    results = c.parse_results()

    method = f'{opt_method}//{settings.method}/{settings.basis_set}'
    homo_calibrated = calibrate_homo(
        homo=results.homo * tools.EV_PER_HARTREE, method=method)
    lumo_calibrated = calibrate_lumo(
        lumo=results.lumo * tools.EV_PER_HARTREE, method=method)
    report = dict(
        path=str(c.directory),
        method=method,
        homo_calibrated=homo_calibrated,
        lumo_calibrated=lumo_calibrated,
        time_seconds=t_end-t_start,
        homo_uncalibrated=results.homo,
        lumo_uncalibrated=results.lumo,
    )

    return report


def cep_calculation_suite(smiles: str, num_processors: int) -> List[dict]:

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
            results = compute_calibrated_homo_lumo(
                atoms=conf,
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
            optimize_structure(atoms=conf, charge=charge,
                               spin_multiplicity=spin_multiplicity,
                               settings=opt_settings))

    # Run QC single point calculations on QC optimized structures
    settings_list = [
        CalculationSettings(method=method, basis_set='def2-SVP',
                            aux_basis_set='def2/J',
                            num_processors=num_processors)
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
            results = compute_calibrated_homo_lumo(atoms=conf,
                                                   charge=charge,
                                                   spin_multiplicity=spin_multiplicity,
                                                   settings=settings,
                                                   opt_method=opt_method)
            reports.append(results)

    return reports
