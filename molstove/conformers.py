from typing import Tuple, List

from rdkit.Chem import Mol, AllChem
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeProperties, MMFFGetMoleculeForceField
from rdkit.Chem.rdMolAlign import AlignMol


def generate_conformers(mol: Mol, max_num_conformers: int, seed: int = 42) -> int:
    """
    Generate conformers for Molecule mol and return number of conformers generated.

    :param mol: molecule for which conformers are generated
    :param max_num_conformers: maximum number of conformers to be generated
    :param seed: random seed of EDG algorithm
    :return: number of conformers generated
    """
    ids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=max_num_conformers,
        maxAttempts=1000,
        pruneRmsThresh=0.1,
        clearConfs=True,
        randomSeed=seed,
        numThreads=0,
    )
    return len(ids)


def minimize_conformers(mol: Mol) -> List[float]:
    """
    Minimize conformers in molecule and return their energies (in kcal/mol).

    :param mol: molecule containing conformers to be optimized
    :return: list of energies in kcal/mol
    """
    energies = []
    props = MMFFGetMoleculeProperties(mol)
    for i in range(mol.GetNumConformers()):
        potential = MMFFGetMoleculeForceField(mol, props, confId=i)
        potential.Minimize()
        energy = potential.CalcEnergy()
        energies.append(energy)

    return energies


def collect_clusters(mol: Mol,
                     energies: List[float],
                     rmsd_threshold=0.2,
                     delta_e_threshold=0.5,
                     energy_window=5,
                     max_num_conformers=10) -> List[AllChem.Conformer]:
    """
    Collect unique conformers of molecule.

    :param mol: molecule containing conformers
    :param energies: energies of the conformers
    :param rmsd_threshold: minimum RMSD between two conformers (in Angstrom)
    :param delta_e_threshold: minimum energy difference between two conformers (in kcal/mol)
    :param energy_window: maximum energy difference between a conformer and the most stable conformer (in kcal/mol)
    :param max_num_conformers: maximum number of conformers
    :return: list of conformers
    """
    assert (mol.GetNumConformers() == len(energies))

    mol_no_h = AllChem.RemoveHs(mol)

    conf_energy_list = sorted([(conf_id, energy) for conf_id, energy in enumerate(energies)], key=lambda t: t[1])
    min_energy = min(energy for conf_id, energy in conf_energy_list)

    accepted_list: List[Tuple[int, float]] = []

    for conf_id, energy in conf_energy_list:
        duplicated = False

        if len(accepted_list) >= max_num_conformers:
            break

        # If conformer too high in energy, break
        if energy - min_energy > energy_window:
            break

        for accepted_id, accepted_energy in accepted_list:
            # If energy is too similar, structures are likely to be similar as well
            if energy - accepted_energy < delta_e_threshold:
                duplicated = True
                break

            rmsd = AlignMol(mol_no_h, mol_no_h, prbCid=conf_id, refCid=accepted_id)
            if rmsd < rmsd_threshold:
                duplicated = True
                break

        if not duplicated:
            accepted_list.append((conf_id, energy))

    return [mol.GetConformer(conf_id) for conf_id, energy in accepted_list]
