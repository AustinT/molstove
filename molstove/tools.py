import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Union, Sequence, Tuple
import uuid
from pathlib import Path

import numpy as np
from rdkit.Chem import Mol, AllChem

ANGSTROM_PER_BOHR = 0.529177  # Angstrom / Bohr
BOHR_PER_ANGSTROM = 1 / ANGSTROM_PER_BOHR  # Bohr / Angstrom

EV_PER_HARTREE = 27.21138602  # eV / Hartree
HARTREE_PER_EV = 1 / EV_PER_HARTREE  # Hartree / eV
CALC_DIR_NAME = "orca-calc"


@dataclass
class Atom:
    element: str
    x: float  # Angstrom
    y: float  # Angstrom
    z: float  # Angstrom


Atoms = List[Atom]


@dataclass
class Orbital:
    occupation: float
    energy: float  # Hartree


def mol_from_smiles(smiles: str) -> Mol:
    """
    Convert SMILES string to Molecule.

    :param smiles: SMILES string
    :return: Molecule
    """
    mol = AllChem.MolFromSmiles(smiles)
    if not mol:
        raise RuntimeError('Failed to generate molecule')

    return AllChem.AddHs(mol)


def get_molecular_charge(mol: Mol) -> int:
    """
    Return molecular charge.

    :param mol: molecule
    :return: molecular charge
    """
    return AllChem.GetFormalCharge(mol)


def conformer_to_atoms(mol: AllChem.Mol, conformer: AllChem.Conformer) -> Atoms:
    """
    Convert RDKit conformer to Atoms.

    :param mol: RDKit molecule of molecule
    :param conformer: conformer to be converted to a list of atoms
    :return: list of atoms
    """
    atoms = []

    pt = AllChem.GetPeriodicTable()
    for atom, position in zip(mol.GetAtoms(), conformer.GetPositions()):
        # position in Angstrom
        symbol = pt.GetElementSymbol(atom.GetAtomicNum())
        atoms.append(Atom(symbol, position[0], position[1], position[2]))

    return atoms


def write_to_json(d: Union[dict, Sequence], path: str) -> None:
    with open(path, mode='w') as f:
        json.dump(d, fp=f, indent=4)  # Indent to make human-readable


def create_tmp_dir_name(include_base_dir=True) -> Path:
    """
    Make a temporary dir name, based on a random string and the date
    """
    date_str = datetime.now().strftime('%Y%m%dT%H%M%S')
    random_hex_str = uuid.uuid4().hex
    tmp_dir_name = Path(random_hex_str + "_" + date_str)
    if include_base_dir:
        return Path(CALC_DIR_NAME) / tmp_dir_name
    return tmp_dir_name


def get_homo_lumo_energies(orbitals: List[Orbital], is_open_shell: bool) -> Tuple[float, float]:
    homo, lumo = 0.0, 0.0
    occupied = 1.0 if is_open_shell else 2.0
    for i, orbital in enumerate(orbitals):
        if np.isclose(orbital.occupation, occupied):
            homo = orbital.energy
        elif np.isclose(orbital.occupation, 0.0):
            lumo = orbital.energy
            break
        else:
            raise RuntimeError(f'Cannot determine if HOMO or LUMO ({orbital.energy})')

    return homo, lumo
