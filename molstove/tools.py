import json
from typing import List, Union, Sequence

from rdkit.Chem import Mol, AllChem

ANGSTROM_PER_BOHR = 0.529177  # Angstrom / Bohr
BOHR_PER_ANGSTROM = 1 / ANGSTROM_PER_BOHR  # Bohr / Angstrom

EV_PER_HARTREE = 27.21138602  # eV / Hartree
HARTREE_PER_EV = 1 / EV_PER_HARTREE  # Hartree / eV


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


def conformer_to_atoms(mol: AllChem.Mol, conformer: AllChem.Conformer) -> List[List]:
    """
    Convert RDKit conformer to a list of atoms for PySCF (in Angstrom).

    :param mol: RDKit molecule of molecule
    :param conformer: conformer to be converted to a list of atoms
    :return: list of atoms
    """
    atoms = []

    pt = AllChem.GetPeriodicTable()
    for atom, position in zip(mol.GetAtoms(), conformer.GetPositions()):
        symbol = pt.GetElementSymbol(atom.GetAtomicNum())
        atoms.append([symbol, tuple(position)])

    return atoms


def write_to_json(d: Union[dict, Sequence], path: str) -> None:
    with open(path, mode='w') as f:
        json.dump(d, fp=f)
